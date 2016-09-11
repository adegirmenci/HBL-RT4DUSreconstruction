#ifndef FRAMESERVERTHREAD_H
#define FRAMESERVERTHREAD_H

#include <QObject>
#include <QMutex>
#include <QMutexLocker>
#include <QThread>
#include <QString>
#include <QTime>
#include <QTimer>
#include <QDebug>
#include <QSharedPointer>

#include <QNetworkInterface>
#include <QTcpServer>
#include <QTcpSocket>
#include <QDataStream>
#include <QByteArray>
#include <QHostAddress>

#include <QQuaternion>
#include <QVector3D>

#include <vector>
#include <memory>

#include "../AscensionWidget/icebot_definitions.h"
//#include "../AscensionWidget/3DGAPI/ATC3DG.h"
//#include "../ICEbot_QT_v1/frmgrabthread.h"

struct FrameExtd{
    QString image_; /*!< Image data. */
    QQuaternion EMq_; /*!< EM reading - rotation as Quaternion. */
    QVector3D EMv_; /*!< EM reading - translation. */
    float phaseHR_; /*!< Phase in the heart cycle (0.0 - 1.0). */
    float phaseResp_; /*!< Phase in the respiration cycle (0.0 - 1.0). */
    qint64 timestamp_; /*!< Timestamp, msec since some epoch. */
    int index_; /*!< Index value of Frame, indicate order of acquisition. */
    QString mask_; /*!< Image mask. */

    //! Constructor.
    explicit FrameExtd(QString img = QString(),
                       QQuaternion emq = QQuaternion(),
                       QVector3D emv = QVector3D(),
                       qint64 ts = -1,
                       float pHR = 0.f,
                       float pResp = 0.f,
                       int id = -1,
                       QString mask = QString()) :
        timestamp_(ts), phaseHR_(pHR), phaseResp_(pResp), index_(id)
    {
        image_ = img;
        EMq_ = emq;
        EMv_ = emv;
        mask_ = mask;
    }
    //! Destructor
    ~FrameExtd() {
    }
};

Q_DECLARE_METATYPE(FrameExtd)

class FrameServerThread : public QObject
{
    Q_OBJECT
public:
    explicit FrameServerThread(QObject *parent = 0);
    ~FrameServerThread();

    friend QDataStream & operator << (QDataStream &o, const FrameExtd& f);
    friend QDataStream & operator >> (QDataStream &i, FrameExtd& f);

signals:
    void statusChanged(int event);
    void finished(); // emit upon termination
    void tcpError(QAbstractSocket::SocketError error);
    void newFrameReceived(FrameExtd frame); // change this to Frame type

public slots:
    void setEpoch(const QDateTime &datetime); // set Epoch
    void initializeFrameServer();
    void startServer();
    void stopServer();
    void newConnectionAvailable();
    void readFrame();
    void handleTcpError(QAbstractSocket::SocketError error);
    const QHostAddress getServerAddress() { return m_serverAddress; }
    const quint16 getServerPort() { return m_serverPort; }

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
    bool m_keepStreaming;

    // Flag to abort actions (e.g. initialize, acquire, etc.)
    bool m_abort;

    int m_frameCount; // keep a count of number of acquired frames

    // server info
    QHostAddress m_serverAddress;
    quint16 m_serverPort;

    QTcpServer *m_TcpServer;
    QTcpSocket *m_TcpSocket;

    FrameExtd m_currExtdFrame;
};

#endif // FRAMESERVERTHREAD_H
