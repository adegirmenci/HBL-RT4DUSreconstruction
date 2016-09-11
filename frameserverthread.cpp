#include "frameserverthread.h"

FrameServerThread::FrameServerThread(QObject *parent) : QObject(parent)
{
    qRegisterMetaType< FrameExtd >("FrameExtd");

    m_isEpochSet = false;
    m_isReady = false;
    m_keepStreaming = false;
    m_frameCount = 0;
    m_abort = false;
    m_serverAddress = QHostAddress(QHostAddress::LocalHost);
    m_serverPort = (quint16)4417;

    m_TcpServer = Q_NULLPTR;
    m_TcpSocket = Q_NULLPTR;

    m_mutex = new QMutex(QMutex::Recursive);
}

FrameServerThread::~FrameServerThread()
{
    m_mutex->lock();
    m_abort = true;
    m_isReady = false;
    m_keepStreaming = false;
    stopServer();
    m_mutex->unlock();

    qDebug() << "Ending FrameServerThread - ID: " << QThread::currentThreadId() << ".";

    delete m_mutex;
    emit finished();
}

void FrameServerThread::initializeFrameServer()
{
    QMutexLocker locker(m_mutex);

    m_TcpServer = new QTcpServer(this);
    //m_TcpServer->setMaxPendingConnections(1);

    connect(m_TcpServer, SIGNAL(newConnection()),
            this, SLOT(newConnectionAvailable()));
    connect(m_TcpServer, SIGNAL(acceptError(QAbstractSocket::SocketError)),
            this, SLOT(handleTcpError(QAbstractSocket::SocketError)));
    connect(this, SIGNAL(tcpError(QAbstractSocket::SocketError)),
            this, SLOT(handleTcpError(QAbstractSocket::SocketError)));

    m_isReady = true;
}

void FrameServerThread::startServer()
{
    QMutexLocker locker(m_mutex);

    if(!m_isReady)
        initializeFrameServer();

    if(!m_TcpServer->listen(m_serverAddress, m_serverPort))
    {
        emit tcpError(m_TcpServer->serverError());
//        qDebug() << tr("FrameServerThread: Unable to start the server: %1.")
//                    .arg(m_TcpServer->errorString());
        emit statusChanged(FRMSRVR_START_FAILED);
    }
    else
    {
        emit statusChanged(FRMSRVR_STARTED);
    }

}

void FrameServerThread::stopServer()
{
    QMutexLocker locker(m_mutex);

    //m_TcpSocket->flush();
    //m_TcpSocket->close();
    if(m_TcpServer)
    {
        m_TcpServer->close();

        if(m_TcpServer->isListening())
            emit statusChanged(FRMSRVR_CLOSE_FAILED);
        else
            emit statusChanged(FRMSRVR_CLOSED);
    }
    else
        emit statusChanged(FRMSRVR_CLOSED);
}

void FrameServerThread::newConnectionAvailable()
{
    QMutexLocker locker(m_mutex);

    emit statusChanged(FRMSRVR_NEW_CONNECTION);

    m_TcpSocket = m_TcpServer->nextPendingConnection();
    connect(m_TcpSocket, SIGNAL(readyRead()), this, SLOT(readFrame()));
    connect(m_TcpSocket, SIGNAL(error(QAbstractSocket::SocketError)),
            this, SLOT(handleTcpError(QAbstractSocket::SocketError)));
    connect(m_TcpSocket, SIGNAL(disconnected()),
            m_TcpSocket, SLOT(deleteLater()));

    if(m_TcpSocket->isReadable())
    {
        qDebug() << "FrameServerThread: Socket is readable.";
    }
    else {
        emit statusChanged(FRMSRVR_SOCKET_NOT_READABLE); }

}

void FrameServerThread::readFrame()
{
    QDataStream in(m_TcpSocket);
    in.setVersion(QDataStream::Qt_5_7);

    quint16 blockSize = 0;

    if (blockSize == 0) {
        if (m_TcpSocket->bytesAvailable() < (int)sizeof(quint16))
            return;
        in >> blockSize;
    }

    if (in.atEnd())
        return;

    FrameExtd nextFrame;
    in >> nextFrame;

//    if (nextFrame == m_currentFrame) {
//        return;
//    }

    m_currExtdFrame = nextFrame;
    emit newFrameReceived(m_currExtdFrame);
    qDebug() << "Received: " << m_currExtdFrame.image_;
    emit FRMSRVR_FRAME_RECEIVED;
}

void FrameServerThread::handleTcpError(QAbstractSocket::SocketError error)
{
    QMutexLocker locker(m_mutex);

    QString errStr;
    switch(error)
    {
    case QAbstractSocket::ConnectionRefusedError:
        errStr = "ConnectionRefusedError"; break;
    case QAbstractSocket::RemoteHostClosedError:
        errStr = "RemoteHostClosedError"; break;
    case QAbstractSocket::HostNotFoundError:
        errStr = "HostNotFoundError"; break;
    case QAbstractSocket::SocketAccessError:
        errStr = "SocketAccessError"; break;
    case QAbstractSocket::SocketResourceError:
        errStr = "SocketResourceError"; break;
    case QAbstractSocket::SocketTimeoutError:
        errStr = "SocketTimeoutError"; break;
    case QAbstractSocket::DatagramTooLargeError:
        errStr = "DatagramTooLargeError"; break;
    case QAbstractSocket::NetworkError:
        errStr = "NetworkError"; break;
    case QAbstractSocket::AddressInUseError:
        errStr = "AddressInUseError"; break;
    case QAbstractSocket::SocketAddressNotAvailableError:
        errStr = "SocketAddressNotAvailableError"; break;
    case QAbstractSocket::UnsupportedSocketOperationError:
        errStr = "UnsupportedSocketOperationError"; break;
    case QAbstractSocket::OperationError:
        errStr = "OperationError"; break;
    case QAbstractSocket::TemporaryError:
        errStr = "TemporaryError"; break;
    case QAbstractSocket::UnknownSocketError:
        errStr = "UnknownSocketError"; break;
    default:
        errStr = "UnknownError";
    }

    qDebug() << tr("Error in FrameServerThread: %1.")
                   .arg(errStr);
}

void FrameServerThread::setEpoch(const QDateTime &datetime)
{
    QMutexLocker locker(m_mutex);
    if(!m_keepStreaming)
    {
        m_epoch = datetime;
        m_isEpochSet = true;

//        emit logEventWithMessage(SRC_FRMSRVR, LOG_INFO, QTime::currentTime(), FRMSRVR_EPOCH_SET,
//                                 m_epoch.toString("yyyy/MM/dd - hh:mm:ss.zzz"));
    }
//    else
//        emit logEvent(SRC_FRMSRVR, LOG_INFO, QTime::currentTime(), FRMSRVR_EPOCH_SET_FAILED);
}

QDataStream & operator << (QDataStream &o, const FrameExtd& f)
{
    return o << f.image_
             << f.index_
             << f.timestamp_
             << f.EMq_
             << f.EMv_
             << f.mask_
             << f.phaseHR_
             << f.phaseResp_
             << '\0';
}

QDataStream & operator >> (QDataStream &i, FrameExtd& f)
{
     i >> f.image_
         >> f.index_
         >> f.timestamp_
         >> f.EMq_
         >> f.EMv_
         >> f.mask_
         >> f.phaseHR_
         >> f.phaseResp_;

    return i;
}
