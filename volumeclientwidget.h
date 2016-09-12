#ifndef VOLUMECLIENTWIDGET_H
#define VOLUMECLIENTWIDGET_H

#include <QWidget>
#include <QThread>
#include <QString>
#include <QTime>
#include <QTimer>
#include <QDebug>
#include <QDir>

#include <QNetworkInterface>
#include <QTcpSocket>
#include <QDataStream>
#include <QByteArray>
#include <QHostAddress>

#include "../AscensionWidget/icebot_definitions.h"

namespace Ui {
class VolumeClientWidget;
}

class VolumeClientWidget : public QWidget
{
    Q_OBJECT

public:
    explicit VolumeClientWidget(QWidget *parent = 0);
    ~VolumeClientWidget();

signals:
    void clientStatusChanged(int status);
    void tcpError(QAbstractSocket::SocketError error);
    void tellVolumeToSave();

private slots:
    void handleClientStatusChanged(int status);
    void setEpoch(const QDateTime &datetime); // set Epoch
    void sendVolume(QString vol);
    void handleTcpError(QAbstractSocket::SocketError error);
    void connectedToHost();
    void disconnectedFromHost();
    const QHostAddress getServerAddress() { return m_serverAddress; }
    const quint16 getServerPort() { return m_serverPort; }

    void on_sendVolumeButton_clicked();

private:
    Ui::VolumeClientWidget *ui;

    // Epoch for time stamps
    // During initializeFrameClient(), check 'isEpochSet' flag
    // If Epoch is set externally from MainWindow, the flag will be true
    // Otherwise, Epoch will be set internally
    QDateTime m_epoch;
    bool m_isEpochSet;

    // Flag to indicate if Frame Client is ready
    // True if initializeFrameClient was successful
    bool m_isReady;

    // Flag to tell that we are still streaming
    bool m_keepStreaming;

    // Flag to abort actions (e.g. initialize, acquire, etc.)
    bool m_abort;

    int m_volumeCount; // keep a count of number of transmitted volumes

    // server info
    QHostAddress m_serverAddress;
    quint16 m_serverPort;

    QTcpSocket *m_TcpSocket;

    QString m_currVolume;
};

#endif // VOLUMECLIENTWIDGET_H
