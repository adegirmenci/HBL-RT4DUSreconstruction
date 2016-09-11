#include "Volume3D.h"

int Volume3D::ms_nVolumes = 0;

Volume3D::Volume3D(QObject *parent) : QObject(parent), m_nFrames(0)
{
	m_volumeID = ms_nVolumes;
	ms_nVolumes++;

	m_timeStart = std::chrono::system_clock::now();

	// GPU arrays
	dev_observations = NULL;
	dev_discretizedObservations = NULL;
	dev_obsIndices = NULL;
	dev_volume = NULL;

	// for the comparison in computeBoundingBox() to work on the first run, need to set these defaults
	m_boundingBoxMin.x = std::numeric_limits<float>::infinity();
	m_boundingBoxMin.y = std::numeric_limits<float>::infinity();
	m_boundingBoxMin.z = std::numeric_limits<float>::infinity();
	m_boundingBoxMax.x = -std::numeric_limits<float>::infinity();
	m_boundingBoxMax.y = -std::numeric_limits<float>::infinity();
	m_boundingBoxMax.z = -std::numeric_limits<float>::infinity();

	// default resolution
	m_gridResolution.x = 1.0f;
	m_gridResolution.y = 1.0f;
	m_gridResolution.z = 1.0f;

	// check if there is a high_resolution_clock and a steady_clock
	std::cout << "high_resolution_clock" << std::endl;
	std::cout << std::chrono::high_resolution_clock::period::num << std::endl;
	std::cout << std::chrono::high_resolution_clock::period::den << std::endl;
	std::cout << "steady = " << std::boolalpha << std::chrono::high_resolution_clock::is_steady << std::endl << std::endl;

	std::cout << "steady_clock" << std::endl;
	std::cout << std::chrono::steady_clock::period::num << std::endl;
	std::cout << std::chrono::steady_clock::period::den << std::endl;
	std::cout << "steady = " << std::boolalpha << std::chrono::steady_clock::is_steady << std::endl << std::endl;

}


Volume3D::~Volume3D()
{
	// free GPU mem
	if (dev_observations)
		cudaFree(dev_observations);
	if (dev_discretizedObservations)
		cudaFree(dev_discretizedObservations);
	if (dev_obsIndices)
		cudaFree(dev_obsIndices);
	if (dev_volume)
		cudaFree(dev_volume);

	ms_nVolumes--;
}

bool Volume3D::addFrame(const Frame &frm)
{
	m_frames.push_back(frm);
	m_nFrames = m_frames.size();

	transformPlane(m_frames.size() - 1);

	return true;
}

void Volume3D::transformPlane(const int idx)
{
	printf("Transforming Plane %d...\n", idx+1);

	double usPlaneLength = 76.6 - 3.9;

	int nRows = m_frames[idx].image_.rows;
	double pixSize = usPlaneLength / static_cast<double>(nRows); // assuming square pixels

	cv::Matx44d T_CT_IMG;
	T_CT_IMG << 0, 1, 0, 0, 0, 0, -1, 0, -1, 0, 0, m_frames[idx].image_.cols / 2.0 * pixSize, 0, 0, 0, 1;
	std::cout << "T_CT_IMG\n" << T_CT_IMG << std::endl;
	printf("Pixel size: %.3f\n", pixSize);

	int2 obsIdxAndLength;
	obsIdxAndLength.x = m_observations.size();
	obsIdxAndLength.y = 0;

	cv::Matx44d premul = m_frames[idx].emData_ * T_CT_IMG;

	//// debug print
	//std::cout << "T_CT_IMG\n" << T_CT_IMG << std::endl;
	//std::cout << "emData_\n" << m_frames[idx].emData_ << std::endl;

	for (int row = 0; row < nRows; row++)
	{
		const unsigned char* maskRow = m_frames[idx].mask_.ptr<unsigned char>(row);

		for (int col = 0; col < m_frames[idx].image_.cols; col++)
		{
			//std::cout << (int)maskRow[col] << std::endl;
			if ((int)maskRow[col]) // not masked
			{
				cv::Matx41d tmpPt(static_cast<double>(col)*pixSize, static_cast<double>(row)*pixSize, 0.0, 1.0);
				cv::Matx41d res = premul * tmpPt;

				float4 tmp;
				tmp.x = (float)res(0);
				tmp.y = (float)res(1);
				tmp.z = (float)res(2);
				tmp.w = 1.0f*(int)m_frames[idx].image_.at<unsigned char>(row, col);

				//// debug print
				//if ((row == nRows / 2) && (col == m_frames[idx].image_.cols / 2))
				//{
				//	std::cout << "tmpPt\n" << tmpPt << std::endl;
				//	std::cout << "res\n" << res << std::endl;
				//}

				//if (col < 5)
				//	printf("%.4f %.4f %.4f\n", tmp.x, tmp.y, tmp.z);

				m_observations.push_back(tmp);
				obsIdxAndLength.y += 1;

				//if (out.size() == 1)
				//{
				//	std::cout << "temp\n" << tmpPt << std::endl;
				//	std::cout << "Result\n" << res << std::endl;
				//}
			}
		}
	}

	m_obsIdxAndLength.push_back(obsIdxAndLength);

	//printf("obsIdxAndLength.x: %d\n", obsIdxAndLength.x);
	//printf("obsIdxAndLength.y: %d\n", obsIdxAndLength.y);
	//printf("m_observations number of elements: %d\n", m_observations.size());

	std::vector<int> v;
	v.push_back(idx);

	updateBoundingBox(v);
	updateGridDimensions();
}

bool Volume3D::removeFrame(unsigned int idx)
{
	// remove observations from vector, and more importantly, from the GPU
	//m_obsIdxAndLength.erase(m_obsIdxAndLength.begin() + idx);

	if (m_frames.size() <= idx)
		return false;

	m_frames.erase(m_frames.begin() + idx);
	m_nFrames = m_frames.size();

	return true;
}

bool Volume3D::computeBoundingBox(unsigned int idx, unsigned int length)
{
	if ((m_observations.size() < idx) || (m_observations.size() < length))
		return false;

	auto xExtremes = std::minmax_element(m_observations.begin() + idx, m_observations.begin() + idx + length,
		[](const float4 &lhs, const float4 &rhs) {
		return lhs.x < rhs.x;
	});

	auto yExtremes = std::minmax_element(m_observations.begin() + idx, m_observations.begin() + idx + length,
		[](const float4 &lhs, const float4 &rhs) {
		return lhs.y < rhs.y;
	});

	auto zExtremes = std::minmax_element(m_observations.begin() + idx, m_observations.begin() + idx + length,
		[](const float4 &lhs, const float4 &rhs) {
		return lhs.z < rhs.z;
	});

	float xmin, ymin, zmin, xmax, ymax, zmax;

	xmin = xExtremes.first->x;
	ymin = yExtremes.first->y;
	zmin = zExtremes.first->z;

	xmax = xExtremes.second->x;
	ymax = yExtremes.second->y;
	zmax = zExtremes.second->z;

	if (m_boundingBoxMin.x > xmin) { m_boundingBoxMin.x = xmin; }
	if (m_boundingBoxMin.y > ymin) { m_boundingBoxMin.y = ymin; }
	if (m_boundingBoxMin.z > zmin) { m_boundingBoxMin.z = zmin; }

	if (m_boundingBoxMax.x < xmax) { m_boundingBoxMax.x = xmax; }
	if (m_boundingBoxMax.y < ymax) { m_boundingBoxMax.y = ymax; }
	if (m_boundingBoxMax.z < zmax) { m_boundingBoxMax.z = zmax; }
	
	return true;
}

bool Volume3D::updateBoundingBox(std::vector<int> idx)
{
	bool res = true;
	for (size_t i = 0; i < idx.size(); i++)
	{
		int2 idxSize = m_obsIdxAndLength[idx[i]];
		res &= computeBoundingBox(idxSize.x, idxSize.y);
	}
	return res;
}

bool Volume3D::updateGridDimensions()
{
	m_gridDimensions.width = next_power_of_two((m_boundingBoxMax.x - m_boundingBoxMin.x) / m_gridResolution.x + 1.0f);
	m_gridDimensions.height = next_power_of_two((m_boundingBoxMax.y - m_boundingBoxMin.y) / m_gridResolution.y + 1.0f);
	m_gridDimensions.depth = next_power_of_two((m_boundingBoxMax.z - m_boundingBoxMin.z) / m_gridResolution.z + 1.0f);

	//m_gridDimensions.width = (int)ceilf((m_boundingBoxMax.x - m_boundingBoxMin.x) / m_gridResolution.x + 1.0f);
	//m_gridDimensions.height = (int)ceilf((m_boundingBoxMax.y - m_boundingBoxMin.y) / m_gridResolution.y + 1.0f);
	//m_gridDimensions.depth = (int)ceilf((m_boundingBoxMax.z - m_boundingBoxMin.z) / m_gridResolution.z + 1.0f);

	printf("Volume Dimensions: %d x %d x %d\n", m_gridDimensions.height, m_gridDimensions.width, m_gridDimensions.depth);

	printf("Bounding Box Min.x %.4f  Max.x %.4f\n", m_boundingBoxMin.x, m_boundingBoxMax.x);
	printf("Bounding Box Min.y %.4f  Max.y %.4f\n", m_boundingBoxMin.y, m_boundingBoxMax.y);
	printf("Bounding Box Min.z %.4f  Max.z %.4f\n", m_boundingBoxMin.z, m_boundingBoxMax.z);

	return true;
}

void Volume3D::calculateGPUparams()
{
	float meanx = (m_boundingBoxMax.x + m_boundingBoxMin.x) / 2.0f;
	float meany = (m_boundingBoxMax.y + m_boundingBoxMin.y) / 2.0f;
	float meanz = (m_boundingBoxMax.z + m_boundingBoxMin.z) / 2.0f;

	float halfLenx = (m_gridDimensions.width / 2)*m_gridResolution.x;
	float halfLeny = (m_gridDimensions.height / 2)*m_gridResolution.y;
	float halfLenz = (m_gridDimensions.depth / 2)*m_gridResolution.z;

	m_boundingBoxMin.x = meanx - halfLenx;
	m_boundingBoxMin.y = meany - halfLeny;
	m_boundingBoxMin.z = meanz - halfLenz;

	m_boundingBoxMax.x = meanx + halfLenx;
	m_boundingBoxMax.y = meany + halfLeny;
	m_boundingBoxMax.z = meanz + halfLenz;

	printf("Volume Dimensions: %d x %d x %d\n", m_gridDimensions.height, m_gridDimensions.width, m_gridDimensions.depth);

	printf("Bounding Box Min.x %.4f  Max.x %.4f\n", m_boundingBoxMin.x, m_boundingBoxMax.x);
	printf("Bounding Box Min.y %.4f  Max.y %.4f\n", m_boundingBoxMin.y, m_boundingBoxMax.y);
	printf("Bounding Box Min.z %.4f  Max.z %.4f\n", m_boundingBoxMin.z, m_boundingBoxMax.z);
}

bool Volume3D::setGridResolution(float newResXYZ)
{
	float3 newRes = make_float3(newResXYZ, newResXYZ, newResXYZ);
	return setGridResolution(newRes);
}

bool Volume3D::setGridResolution(float newResX, float newResY, float newResZ)
{
	float3 newRes = make_float3(newResX, newResY, newResZ);
	return setGridResolution(newRes);
}

bool Volume3D::setGridResolution(float3 newRes)
{
	if ((newRes.x <= 0) || (newRes.y <= 0) || (newRes.z <= 0))
		return false;

	m_gridResolution = newRes;

	updateGridDimensions();

	return true;
}

bool Volume3D::xferHostToDevice()
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	printf("Set CUDA Device to 0.\n");
	m_cudaStatus = cudaSetDevice(0);
	if (m_cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		handleCUDAerror();
		return false;
	}

	// Allocate GPU buffers
	printf("CUDA malloc dev_observations.\n");
	m_cudaStatus = cudaMalloc((void**)&dev_observations, m_observations.size() * sizeof(float4));
	if (m_cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		handleCUDAerror();
		return false;
	}

	printf("CUDA malloc dev_discretizedObservations.\n");
	m_cudaStatus = cudaMalloc((void**)&dev_discretizedObservations, m_observations.size() * sizeof(float4));
	if (m_cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		handleCUDAerror();
		return false;
	}
	
	printf("CUDA malloc dev_obsIndices.\n");
	m_cudaStatus = cudaMalloc((void**)&dev_obsIndices, m_observations.size() * sizeof(int4));
	if (m_cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		handleCUDAerror();
		return false;
	}

	// TODO: use cudaMalloc3D instead to get cudaPitchedPtr
	printf("CUDA malloc dev_volume, %d x %d x %d.\n", m_gridDimensions.width, m_gridDimensions.height, m_gridDimensions.depth);
	int nElems = m_gridDimensions.width * m_gridDimensions.height * m_gridDimensions.depth;
	m_cudaStatus = cudaMalloc((void**)&dev_volume, nElems * sizeof(int));
	if (m_cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		handleCUDAerror();
		return false;
	}

	// set volume to -1
	printf("CUDA memset dev_volume to -1.\n");
	m_cudaStatus = cudaMemset(dev_volume, -1, nElems * sizeof(int));
	if (m_cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		handleCUDAerror();
		return false;
	}

	// Copy input vectors from host memory to GPU buffers.
	printf("CUDA memcpy m_observations to dev_discretizedObservations.\n");
	m_cudaStatus = cudaMemcpy(dev_observations, &(m_observations[0]), m_observations.size() * sizeof(float4), cudaMemcpyHostToDevice);
	if (m_cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		handleCUDAerror();
		return false;
	}

	return true;
}

bool Volume3D::discretizePoints_gpu(const std::vector<int> &idx)
{
	// TODO: need to be able to selectively discretize new, incoming frames

	// call kernel
	printf("CUDA kernel exec: discretizePoints().\n");
	m_cudaStatus = discretizePoints(dev_observations, dev_discretizedObservations, dev_obsIndices, m_observations.size(), m_gridResolution, m_boundingBoxMin, m_boundingBoxMax, m_gridDimensions);
	// Check for any errors launching the kernel
	if (m_cudaStatus != cudaSuccess) {
		fprintf(stderr, "discretizePoints launch failed: %s\n", cudaGetErrorString(m_cudaStatus));
		handleCUDAerror();
		return false;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	printf("CUDA device sync.\n\n");
	m_cudaStatus = cudaDeviceSynchronize();
	if (m_cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching discretizePoints!\n", m_cudaStatus);
		handleCUDAerror();
		return false;
	}

	return true;
}

bool Volume3D::addObsToVolume_gpu(const std::vector<int> &idx)
{
	// TODO: need to be able to selectively discretize new, incoming frames

	// call kernel
	printf("CUDA kernel exec: addObsToVolume().\n");
	m_cudaStatus = addObsToVolume(dev_obsIndices, dev_volume, m_observations.size(), m_gridDimensions);
	// Check for any errors launching the kernel
	if (m_cudaStatus != cudaSuccess) {
		fprintf(stderr, "addObsToVolume launch failed: %s\n", cudaGetErrorString(m_cudaStatus));
		handleCUDAerror();
		return false;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	printf("CUDA device sync.\n\n");
	m_cudaStatus = cudaDeviceSynchronize();
	if (m_cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addObsToVolume!\n", m_cudaStatus);
		handleCUDAerror();
		return false;
	}

	return true;
}

bool Volume3D::computeConvHull()
{
	// resize volume
	int nElems = m_gridDimensions.width * m_gridDimensions.height * m_gridDimensions.depth;
	if (m_volume.size() != nElems)
	{
		m_volume.resize(nElems);
		printf("Resizing m_volume to accomodate CUDA array dev_volume.\n");
	}

	// transfer volume
	printf("CUDA memcpy dev_volume to m_volume, %d x %d x %d.\n", m_gridDimensions.width, m_gridDimensions.height, m_gridDimensions.depth);
	m_cudaStatus = cudaMemcpy(&(m_volume[0]), dev_volume, nElems * sizeof(int), cudaMemcpyDeviceToHost);
	if (m_cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		handleCUDAerror();
		return false;
	}

	// get non-negative elements and put into Qhull
	printf("\nRun QHull.\n");
	m_convHullVerts.clear();
	size_t linIdx = 0;
	for (size_t k = 0; k < m_gridDimensions.depth; k++)
	{
		for (size_t j = 0; j < m_gridDimensions.height; j++)
		{
			for (size_t i = 0; i < m_gridDimensions.width; i++)
			{
				if (m_volume[linIdx] != -1)
				{
					orgQhull::float3qhull tmp;
					tmp.x = (float)i;
					tmp.y = (float)j;
					tmp.z = (float)k;
					m_convHullVerts.push_back(tmp);
				}
				linIdx++;
			}
		}
	}
	
	qhull.runQhull3D(m_convHullVerts, "Qt");

	return true;
}

bool Volume3D::fillConvHull()
{
	//std::vector< std::pair<orgQhull::float3qhull, double> > facetsNormals;
	//for each (QhullFacet facet in qhull.facetList().toStdVector())
	//{
	//	if (facet.hyperplane().isValid())
	//	{
	//		auto coord = facet.hyperplane().coordinates();
	//		orgQhull::float3qhull normal;
	//		normal.x = coord[0]; normal.y = coord[1]; normal.z = coord[2];
	//		double offset = facet.hyperplane().offset();
	//		facetsNormals.push_back(std::pair<orgQhull::float3qhull, double>(normal, offset));
	//	}
	//}

	//QhullPoint center = qhull.origin();

	printf("Compute inhull.\n");
	QhullPoint currPoint = qhull.origin(); // using origin as a dummy point to initialize
	//std::vector<int> isInside(qhull.facetCount(), 0);
	std::vector<QhullFacet> hullFacets = qhull.facetList().toStdVector();
	size_t linIdx = 0;
	for (size_t k = 0; k < m_gridDimensions.depth; k++)
	{
		printf(".");
		for (size_t j = 0; j < m_gridDimensions.height; j++)
		{
			for (size_t i = 0; i < m_gridDimensions.width; i++)
			{
				currPoint[0] = i; currPoint[1] = j; currPoint[2] = k;

				pointT *point = currPoint.getBaseT();
				realT dist;
				int sum = 0;
				for each (QhullFacet facet in hullFacets)
				{
					qh_distplane(qhull.qh(), point, facet.getBaseT() , &dist);
					if (dist < (qhull.qh()->min_vertex - 2 * qhull.qh()->DISTround))
					{
						//isInside[idx] = 1;
						sum++;
					}
				}
				if (sum != qhull.facetCount()) // point is outside hull
					m_volume[linIdx] = 0; //set to zero to establish the interpolation boundary	

				linIdx++;
			}
		}
	}
	printf("\n");

	// transfer volume
	int nElems = m_gridDimensions.width * m_gridDimensions.height * m_gridDimensions.depth;
	printf("CUDA memcpy m_volume to dev_volume.\n");
	m_cudaStatus = cudaMemcpy(dev_volume, &(m_volume[0]), nElems * sizeof(int), cudaMemcpyHostToDevice);
	if (m_cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		handleCUDAerror();
		return false;
	}

	return true;
}

bool Volume3D::xferDeviceToHost()
{
	if (m_discretizedObservations.size() != m_observations.size())
	{
		m_discretizedObservations.resize(m_observations.size());
		printf("Resizing m_discretizedObservations to accomodate CUDA array dev_discretizedObservations.\n");
	}
	
	// Copy output vector from GPU buffer to host memory.
	printf("CUDA memcpy dev_discretizedObservations to m_discretizedObservations.\n");
	m_cudaStatus = cudaMemcpy(&(m_discretizedObservations[0]), dev_discretizedObservations, m_observations.size() * sizeof(float4), cudaMemcpyDeviceToHost);
	if (m_cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		handleCUDAerror();
		return false;
	}

	if (m_obsIndices.size() != m_observations.size())
	{
		m_obsIndices.resize(m_observations.size());
		printf("Resizing m_obsIndices to accomodate CUDA array dev_discretizedObservations.\n");
	}

	printf("CUDA memcpy dev_obsIndices to m_discretizedObservations.\n");
	m_cudaStatus = cudaMemcpy(&(m_obsIndices[0]), dev_obsIndices, m_observations.size() * sizeof(int4), cudaMemcpyDeviceToHost);
	if (m_cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		handleCUDAerror();
		return false;
	}

	int nElems = m_gridDimensions.width * m_gridDimensions.height * m_gridDimensions.depth;
	if (m_volume.size() != nElems)
	{
		m_volume.resize(nElems);
		printf("Resizing m_volume to accomodate CUDA array dev_volume.\n");
	}

	printf("CUDA memcpy dev_volume to m_volume, %d x %d x %d.\n", m_gridDimensions.width, m_gridDimensions.height, m_gridDimensions.depth);
	m_cudaStatus = cudaMemcpy(&(m_volume[0]), dev_volume, nElems * sizeof(int), cudaMemcpyDeviceToHost);
	if (m_cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		handleCUDAerror();
		return false;
	}

	// example output
	printf("Obs     %d: %.2f, %.2f, %.2f, %.2f\n", 0, m_observations[0].x, m_observations[0].y, m_observations[0].z, m_observations[0].w);
	printf("DiscObs %d: %.2f, %.2f, %.2f, %.2f\n", 0, m_discretizedObservations[0].x, m_discretizedObservations[0].y, m_discretizedObservations[0].z, m_discretizedObservations[0].w);
	printf("Indices %d: %d  , %d  , %d  , %d  \n", 0, m_obsIndices[0].x, m_obsIndices[0].y, m_obsIndices[0].z, m_obsIndices[0].w);

	return true;
}

bool Volume3D::saveVolumeToDisk()
{
	auto now = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t(now);
	std::stringstream ss;
	ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S") << "_vol_1.raw";

	return saveVolumeToDisk(ss.str());
}

bool Volume3D::saveVolumeToDisk(std::string fname)
{
    // Open as ***BINARY*** file for output
    // It took about 15 hours of debugging to realize the 'b' was missing ...
    FILE *fp = fopen(fname.c_str(), "wb");

	if(!fp)
	{
		fprintf(stderr, "Error opening file '%s'\n", fname.c_str());
		return false;
	}

    // convert to char
    std::vector<uchar> temp;
    temp.resize(m_volume.size());
    for (size_t i = 0; i < temp.size(); i++)
    {
        if (m_volume[i] < 0)
            m_volume[i] = 0;
        else if (m_volume[i] > 255)
            m_volume[i] = 255;
        temp[i] = (uchar)m_volume[i];
    }

    //size_t idx = 0;
    //for (size_t k = 0; k < m_gridDimensions.depth; k++)
    //{
    //	for (size_t i = 0; i < m_gridDimensions.width; i++)
    //	{
    //		for (size_t j = 0; j < m_gridDimensions.height; j++)
    //		{
    //			size_t linidx = k*m_gridDimensions.height*m_gridDimensions.width + j*m_gridDimensions.width + i;
    //			//size_t linidx = k*m_gridDimensions.height*m_gridDimensions.width + j*m_gridDimensions.height + i;
    //			//size_t linidx = k*m_gridDimensions.height*m_gridDimensions.width + i*m_gridDimensions.height + j;
    //			int pixVal = m_volume[linidx];
    //			if (pixVal < 0)
    //			{	pixVal = 0;     }
    //			else if (pixVal > 255)
    //			{   pixVal = 255;	}
    //			temp[idx] = (uchar)pixVal;
    //			idx++;
    //		}
    //	}
    //}

    //for (size_t i = 0; i < m_volume.size(); i++)
    //{
    //	size_t d = i / (m_gridDimensions.width* m_gridDimensions.height);
    //	size_t h = (i - d*m_gridDimensions.width* m_gridDimensions.height) / m_gridDimensions.height;
    //	size_t w = i - d*m_gridDimensions.width* m_gridDimensions.height - h*m_gridDimensions.height;
    //	int pixVal = m_volume[i];
    //	if (pixVal < 0)
    //	{
    //		pixVal = 0;
    //		//pixVal = ((d + h + w) * 255) / (m_gridDimensions.width + m_gridDimensions.height + m_gridDimensions.depth);
    //	}
    //	else if (pixVal > 255)
    //	{
    //		pixVal = 255;
    //	}
    //	temp[d*m_gridDimensions.width* m_gridDimensions.height + h*m_gridDimensions.width + w] = (uchar)pixVal;
    //}

	// write as char
	//size_t written = fwrite((void *)temp.data(), sizeof(std::vector<uchar>::value_type), temp.size(), fp);
	size_t written = fwrite((void *)&temp[0], sizeof(uchar), temp.size(), fp);

	// write as integer
	//size_t written = fwrite(m_volume.data(), sizeof m_volume[0], m_volume.size(), fp);

	fclose(fp);

	printf("Written '%s', %d bytes\n", fname.c_str(), written);

	return true;
}

void Volume3D::handleCUDAerror()
{
	cudaFree(dev_observations);
	cudaFree(dev_discretizedObservations);
	cudaFree(dev_obsIndices);
	cudaFree(dev_volume);
}
