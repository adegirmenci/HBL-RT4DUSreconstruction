#ifndef RT3DRECONSTWORKER_H
#define RT3DRECONSTWORKER_H

#include <QObject>
#include <QMutex>
#include <QMutexLocker>
#include <QThread>
#include <QString>
#include <QTime>
#include <QTimer>
#include <QDebug>
#include <QSharedPointer>
#include <QDir>
#include <QMatrix4x4>

#include <vector>
#include <memory>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Volume3D.h"
#include "frameserverthread.h"

class RT3DReconstWorker : public QObject
{
    Q_OBJECT
public:
    explicit RT3DReconstWorker(QObject *parent = 0);
    ~RT3DReconstWorker();

signals:
    void statusChanged(int event);
    void finished(); // emit upon termination
    void volumeSaved(QString vol);

public slots:
    void setEpoch(const QDateTime &datetime); // set Epoch
    void initializeWorker();
    void addFrame(FrameExtd frm);
    void loadMask(QString &path, cv::Mat &mask, cv::Rect &roi);
    void interpolate();
    void tellVolumeToSave();
    void passthroughVolumeSaved(QString vol);

private:
    // Instead of using "m_mutex.lock()"
    // use "QMutexLocker locker(&m_mutex);"
    // this will unlock the mutex when the locker goes out of scope
    mutable QMutex *m_mutex;

    // Epoch for time stamps
    // During initializeFrameServer(), check 'isEpochSet' flag
    // If Epoch is set externally from MainWindow, the flag will be true
    // Otherwise, Epoch will be set internally
    QDateTime m_epoch;
    bool m_isEpochSet;

    // Flag to indicate if Frame Server is ready
    // True if initializeFrameServer was successful
    bool m_isReady;

    // Flag to tell that we are still streaming
    bool m_keepReconstructing;

    // Flag to abort actions (e.g. initialize, acquire, etc.)
    bool m_abort;

    Volume3D *m_volume;
};

#endif // RT3DRECONSTWORKER_H
