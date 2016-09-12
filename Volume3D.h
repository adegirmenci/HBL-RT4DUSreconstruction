#ifndef HBL_VOLUME3D_H
#define HBL_VOLUME3D_H

#include <QObject>
#include <QString>
#include <QDateTime>
#include <QDir>
#include <QDebug>

#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <iostream>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "libqhullcpp/QhullQh.h"
#include "libqhullcpp/QhullPoint.h"
#include "libqhullcpp/QhullError.h"
#include "libqhullcpp/QhullFacet.h"
#include "libqhullcpp/QhullFacetList.h"
#include "libqhullcpp/QhullLinkedList.h"
#include "libqhullcpp/QhullVertex.h"
#include "libqhullcpp/Qhull.h"

using orgQhull::QhullError;
using orgQhull::QhullFacet;
using orgQhull::QhullFacetList;
using orgQhull::QhullQh;
using orgQhull::Qhull;
using orgQhull::QhullPoint;
using orgQhull::QhullVertex;
using orgQhull::QhullVertexSet;

typedef unsigned int  uint;
typedef unsigned char uchar;
typedef unsigned char VolumeType;

extern "C"
cudaError_t discretizePoints(const float4 *dev_points, float4 *dev_discPoints, int4 * dev_pointIdx, const int nPoints, const float3 mapRes, const float3 gridMin, const float3 gridMax, const cudaExtent gridDims);

extern "C"
cudaError_t addObsToVolume(const int4 * dev_pointIdx, int * dev_volume, const int nPoints, const cudaExtent gridDims);

static int next_power_of_two(float a_F){
	if (a_F < 0.0f)
	{
		fprintf(stderr, "Negative number used in power of two function. Flipping sign to positive!\n");
		a_F = -1.0f*a_F;
	}

	// http://stackoverflow.com/a/466392
	int f = *((int*)&a_F);
	int b = (f << 9) != 0; // If we're a power of two this is 0, otherwise this is 1

	f >>= 23; // remove fractional part of floating point number
	f -= 127; // subtract 127 (the bias) from the exponent

	// adds one to the exponent if were not a power of two, 
	// then raises our new exponent to the power of two again.
	return (1 << (f + b));
}

static int next_power_of_two2(float a)
{
	if (a < 0.0f)
	{
		fprintf(stderr, "Negative number used in power of two function. Flipping sign to positive!\n");
		a = -1.0f*a;
	}
	else if (a < 1.0f)
	{
		a = 1.0f;
	}

	unsigned int v = static_cast<unsigned int>(a);

	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

struct Frame{
	cv::Mat image_; /*!< Image data. */
	cv::Matx44d emData_; /*!< EM reading. */
	float phaseHR_; /*!< Phase in the heart cycle (0.0 - 1.0). */
	float phaseResp_; /*!< Phase in the respiration cycle (0.0 - 1.0). */
	time_t timestamp_; /*!< Timestamp, msec since some epoch. */
	int index_; /*!< Index value of Frame, indicate order of acquisition. */
	cv::Mat mask_; /*!< Image mask. */

	//! Constructor.
	explicit Frame(cv::Mat img = cv::Mat(), cv::Matx44d em = cv::Matx44d(), __int64 ts = -1, int id = -1, cv::Mat mask = cv::Mat()) :
		timestamp_(ts), index_(id)
	{
		image_ = img; 
		emData_ = em;
		mask_ = mask;
	}
	//! Destructor
	~Frame() {
		image_.release();
		mask_.release();
	}
};

class Volume3D : public QObject
{
    Q_OBJECT
public:
    explicit Volume3D(QObject *parent = 0);
	~Volume3D();

	bool addFrame(const Frame &frm);
	bool removeFrame(unsigned int idx);
	bool computeBoundingBox(unsigned int idx, unsigned int length);
	bool updateBoundingBox(std::vector<int> idx);
	bool updateGridDimensions();
	void calculateGPUparams();

	bool setGridResolution(float newResXYZ);
	bool setGridResolution(float newResX, float newResY, float newResZ);
	bool setGridResolution(float3 newRes);

	bool xferHostToDevice();
	bool discretizePoints_gpu(const std::vector<int> &idx);
	bool addObsToVolume_gpu(const std::vector<int> &idx);
	bool computeConvHull();
	bool fillConvHull();
	bool xferDeviceToHost();
	bool saveVolumeToDisk();
    bool saveVolumeToDisk(QString fname);

	const int getVolumeNumber() { return m_volumeID; }

signals:
    void logEvent(int eventID);
    void volumeSaved(QString vol);

private:
	size_t m_nFrames; // number of 2D frames contained in this 3D volume
	std::vector<Frame> m_frames; // Frame data
	std::chrono::system_clock::time_point m_timeStart; // time when Volume3D was initialized
	std::chrono::system_clock::time_point m_timeLastUpdate; // time when Volume3D was updated
	float3 m_boundingBoxMin; // min corner of bounding box
	float3 m_boundingBoxMax; // max corner of bounding box
	float3 m_gridResolution; // resolution along each axis
	cudaExtent m_gridDimensions; // dimensions of the 3D grid
	std::vector<float4> m_observations; // location (x,y,z) and intensity (w) of US data
	std::vector<float4> m_discretizedObservations; // discretized location and intensity (can still be a float, if resolution is not 1/integer)
	std::vector<int4> m_obsIndices; // index of observation in the 3D grid
	std::vector<int2> m_obsIdxAndLength; // contains the index of where an observation begins in the m_observations vector, and how many elements it has
	std::vector<int> m_volume;

	int m_volumeID; // ID of this volume
	static int ms_nVolumes; //counter to keep track of number of volumes

	void transformPlane(const int idx);

	// QHull
	Qhull qhull;
	std::vector<orgQhull::float3qhull> m_convHullVerts;

	// GPU
	float4 *dev_observations;
	float4 *dev_discretizedObservations;
	int4 * dev_obsIndices;
	int *dev_volume;
	cudaError_t m_cudaStatus;

	void handleCUDAerror();
};

#endif // HBL_VOLUME3D_H
