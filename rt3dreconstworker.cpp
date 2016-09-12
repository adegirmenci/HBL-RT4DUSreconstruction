#include "rt3dreconstworker.h"

RT3DReconstWorker::RT3DReconstWorker(QObject *parent) : QObject(parent)
{
    //qRegisterMetaType< FrameExtd >("FrameExtd");

    m_isEpochSet = false;
    m_isReady = false;
    m_keepReconstructing = false;
    m_abort = false;

    m_mutex = new QMutex(QMutex::Recursive);
    m_volume = Q_NULLPTR;
}

RT3DReconstWorker::~RT3DReconstWorker()
{
    m_mutex->lock();
    m_abort = true;
    m_isReady = false;
    m_keepReconstructing = false;
    m_volume->deleteLater();
    m_mutex->unlock();

    qDebug() << "Ending RT3DReconstWorker - ID: " << QThread::currentThreadId() << ".";

    delete m_mutex;
    emit finished();
}

void RT3DReconstWorker::initializeWorker()
{
    QMutexLocker locker(m_mutex);

    m_volume = new Volume3D(this->parent());

    m_isReady = true;

    connect(m_volume, SIGNAL(volumeSaved(QString)),
            this, SLOT(passthroughVolumeSaved(QString)));

    printf("RT3DReconstWorker initialized.\n");
}

void RT3DReconstWorker::addFrame(FrameExtd frm)
{
    Frame newFrame;
    cv::Rect roi;
    loadMask(frm.mask_, newFrame.mask_, roi);
    printf("ROI: %d x %d\n", roi.width, roi.height);

    newFrame.image_ = cv::imread((frm.image_+tr(".jp2")).toStdString(), CV_LOAD_IMAGE_GRAYSCALE );
    printf("Frame read: %d x %d\n", newFrame.image_.cols, newFrame.image_.rows);

    //cvSetImageROI(newFrame.image_, roi);
    newFrame.image_ = newFrame.image_(roi);
    printf("Mask applied: %d x %d\n", newFrame.image_.cols, newFrame.image_.rows);

    QMatrix4x4 tform;
    tform.rotate(frm.EMq_);
    tform.translate(frm.EMv_);
    newFrame.emData_ = cv::Matx44f(tform.data());
    cv::transpose(newFrame.emData_,newFrame.emData_);

    for(size_t i = 0; i < 4; i++)
    {
        std::cout << tform.row(i).x() << " "
                  << tform.row(i).y() << " "
                  << tform.row(i).z() << " "
                  << tform.row(i).w()
                  << std::endl;
    }
    std::cout << newFrame.emData_ << std::endl;

    newFrame.index_ = frm.index_;
    newFrame.phaseHR_ = frm.phaseHR_;
    newFrame.phaseResp_ = frm.phaseResp_;
    newFrame.timestamp_ = frm.timestamp_;

    m_volume->addFrame(newFrame);

}

void RT3DReconstWorker::loadMask(QString &path, cv::Mat &mask, cv::Rect &roi)
{
    printf("Reading from: %s\n", path.toStdString().c_str());

    // open file for read
    std::ifstream source(path.toStdString().c_str(), std::ios::in | std::ios::binary);

    if (source)
    {
        int32_t imWidth = 0, imHeight = 0, x = 0, y = 0, w = 0, h = 0;

        int32_t *tmp = (int32_t*)malloc(sizeof(int32_t));

        source.read((char*)tmp, sizeof(int32_t));
        imWidth = *tmp;
        source.read((char*)tmp, sizeof(int32_t));
        imHeight = *tmp;
        source.read((char*)tmp, sizeof(int32_t));
        x = *tmp;
        source.read((char*)tmp, sizeof(int32_t));
        y = *tmp;
        source.read((char*)tmp, sizeof(int32_t));
        w = *tmp;
        source.read((char*)tmp, sizeof(int32_t));
        h = *tmp;

        printf("Image: W: %d, H: %d\n", imWidth, imHeight);
        printf("Mask:  X: %d, Y: %d, W: %d, H: %d\n", x, y, w, h);

        roi = cv::Rect(x, y, w + 1, h + 1);

        mask.create(roi.size(), CV_8UC1);

        for (int row = 0; row < h + 1; row++)
        {
            for (int col = 0; col < w + 1; col++)
            {
                unsigned char &c = mask.at<unsigned char>(row, col);
                source >> c;
            }
        }

        printf("Mask data loaded.\n");

        source.close(); // close file

        free(tmp);
    }
    else
        printf("Problem opening mask file.\n");

}

void RT3DReconstWorker::interpolate()
{
    m_volume->calculateGPUparams();

    m_volume->xferHostToDevice();
    m_volume->discretizePoints_gpu(std::vector<int>());
    m_volume->addObsToVolume_gpu(std::vector<int>());
    //m_volume->computeConvHull();
    //m_volume->fillConvHull(); // TODO: speedup
    //TODO: interpolate
    m_volume->xferDeviceToHost();

    //m_volume->saveVolumeToDisk();
}

void RT3DReconstWorker::tellVolumeToSave()
{
    if(m_volume)
        m_volume->saveVolumeToDisk();
}

void RT3DReconstWorker::passthroughVolumeSaved(QString vol)
{
    emit volumeSaved(vol);
}

void RT3DReconstWorker::setEpoch(const QDateTime &datetime)
{
    QMutexLocker locker(m_mutex);
    if(!m_keepReconstructing)
    {
        m_epoch = datetime;
        m_isEpochSet = true;

//        emit logEventWithMessage(SRC_RECONST, LOG_INFO, QTime::currentTime(), RECONST_EPOCH_SET,
//                                 m_epoch.toString("yyyy/MM/dd - hh:mm:ss.zzz"));
    }
//    else
    //        emit logEvent(SRC_RECONST, LOG_INFO, QTime::currentTime(), RECONST_EPOCH_SET_FAILED);
}
