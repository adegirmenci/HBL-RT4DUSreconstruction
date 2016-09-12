#include "volumeclientwidget.h"
#include "ui_volumeclientwidget.h"

VolumeClientWidget::VolumeClientWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::VolumeClientWidget)
{
    ui->setupUi(this);

    m_isEpochSet = false;
    m_keepStreaming = false;
    m_volumeCount = 0;
    m_abort = false;
    m_serverAddress = QHostAddress(QHostAddress::LocalHost);
    m_serverPort = (quint16)4419;

    m_currVolume = tr("");

    m_TcpSocket = new QTcpSocket(this);

    connect(m_TcpSocket, SIGNAL(error(QAbstractSocket::SocketError)),
            this, SLOT(handleTcpError(QAbstractSocket::SocketError)));
    connect(this, SIGNAL(tcpError(QAbstractSocket::SocketError)),
            this, SLOT(handleTcpError(QAbstractSocket::SocketError)));

    connect(m_TcpSocket, SIGNAL(connected()), this, SLOT(connectedToHost()));
    connect(m_TcpSocket, SIGNAL(disconnected()), this, SLOT(disconnectedFromHost()));

    connect(this, SIGNAL(clientStatusChanged(int)),
            this, SLOT(handleClientStatusChanged(int)));

    m_isReady = true;
}

VolumeClientWidget::~VolumeClientWidget()
{
    m_abort = true;
    m_isReady = false;
    m_keepStreaming = false;
    if(m_TcpSocket)
    {
        m_TcpSocket->flush();
        m_TcpSocket->disconnectFromHost();
        m_TcpSocket->deleteLater();
    }

    qDebug() << "Closing VolumeClientWidget - Thread ID: " << QThread::currentThreadId() << ".";

    delete ui;
}

void VolumeClientWidget::sendVolume(QString vol)
{
    m_currVolume = vol;

    if(m_isReady)
    {
        QByteArray block;
        QDataStream out(&block, QIODevice::WriteOnly);
        out.setVersion(QDataStream::Qt_5_7);
        out << (quint16)0;
        out << m_currVolume;
        out.device()->seek(0);
        out << (quint16)(block.size() - sizeof(quint16));

        m_TcpSocket->connectToHost(m_serverAddress, m_serverPort, QIODevice::WriteOnly);

        if (m_TcpSocket->waitForConnected(250))
        {
              qDebug("Connected!");

            m_TcpSocket->write(block);
            m_TcpSocket->flush();
            m_TcpSocket->disconnectFromHost();
            if( (m_TcpSocket->state() != QAbstractSocket::UnconnectedState) &&
                (!m_TcpSocket->waitForDisconnected(250)) ) {
                emit clientStatusChanged(VOLCLNT_DISCONNECTION_FAILED); }
        }
        else
            emit clientStatusChanged(VOLCLNT_CONNECTION_FAILED);
    }
    else
        emit clientStatusChanged(VOLCLNT_NOT_READY);
}

void VolumeClientWidget::connectedToHost()
{
    emit clientStatusChanged(VOLCLNT_CONNECTED);
}

void VolumeClientWidget::disconnectedFromHost()
{
    emit clientStatusChanged(VOLCLNT_DISCONNECTED);
}

void VolumeClientWidget::handleTcpError(QAbstractSocket::SocketError error)
{
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

    qDebug() << tr("Error in VolumeClientWidget: %1.")
                   .arg(errStr);
}

void VolumeClientWidget::handleClientStatusChanged(int status)
{
    switch(status)
    {
    case VOLCLNT_CONNECTED:
        ui->statusTextEdit->appendPlainText("Connected to server.");
        ui->addrPortLineEdit->setText(tr("%1:%2")
                                      .arg(getServerAddress().toString())
                                      .arg(getServerPort()));
        break;
    case VOLCLNT_CONNECTION_FAILED:
        ui->statusTextEdit->appendPlainText("Server connection failed.");
        break;
    case VOLCLNT_DISCONNECTED:
        ui->statusTextEdit->appendPlainText("Server connection closed.");
        break;
    case VOLCLNT_DISCONNECTION_FAILED:
        ui->statusTextEdit->appendPlainText("Server disconnect failed.");
        break;
    case VOLCLNT_SOCKET_NOT_WRITABLE:
        ui->statusTextEdit->appendPlainText("Incoming connection.");
        break;
    case VOLCLNT_VOLUME_SENT:
        ui->statusTextEdit->appendPlainText("Frame sent.");
        break;
    case VOLCLNT_EPOCH_SET:
        ui->statusTextEdit->appendPlainText("Epoch set.");
        break;
    case VOLCLNT_EPOCH_SET_FAILED:
        ui->statusTextEdit->appendPlainText("Epoch set failed.");
        break;
    case VOLCLNT_NOT_READY:
        ui->statusTextEdit->appendPlainText("Volume client is not ready.");
        break;
    default:
        ui->statusTextEdit->appendPlainText("Unknown worker state.");
    }
}

void VolumeClientWidget::setEpoch(const QDateTime &datetime)
{
    if(!m_keepStreaming)
    {
        m_epoch = datetime;
        m_isEpochSet = true;

//        emit logEventWithMessage(SRC_VOLCLNT, LOG_INFO, QTime::currentTime(), VOLCLNT_EPOCH_SET,
//                                 m_epoch.toString("yyyy/MM/dd - hh:mm:ss.zzz"));
    }
//    else
    //        emit logEvent(SRC_VOLCLNT, LOG_INFO, QTime::currentTime(), VOLCLNT_EPOCH_SET_FAILED);
}



void VolumeClientWidget::on_sendVolumeButton_clicked()
{
    emit tellVolumeToSave();
}
