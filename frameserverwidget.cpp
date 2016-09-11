#include "frameserverwidget.h"
#include "ui_frameserverwidget.h"

FrameServerWidget::FrameServerWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FrameServerWidget)
{
    ui->setupUi(this);

    m_keepServerRunning = false;

    m_worker = new FrameServerThread;
    m_worker->moveToThread(&m_thread);

    connect(&m_thread, SIGNAL(finished()), m_worker, SLOT(deleteLater()));
    m_thread.start();

    connect(m_worker, SIGNAL(statusChanged(int)), this, SLOT(workerStatusChanged(int)));

    connect(this, SIGNAL(startServer()), m_worker, SLOT(startServer()));
    connect(this, SIGNAL(stopServer()), m_worker, SLOT(stopServer()));
    // connect(m_worker, SIGNAL(newFrameReceived(Frame)), ..., SLOT(...(Frame)));
}

FrameServerWidget::~FrameServerWidget()
{
    m_thread.quit();
    m_thread.wait();
    qDebug() << "FrameServerWidget thread quit.";

    delete ui;
}

void FrameServerWidget::workerStatusChanged(int status)
{
    switch(status)
    {
    case FRMSRVR_STARTED:
        ui->toggleServerButton->setText("Stop Server");
        ui->addrPortLineEdit->setText(tr("%1:%2")
                                      .arg(m_worker->getServerAddress().toString())
                                      .arg(m_worker->getServerPort()));
        ui->statusTextEdit->appendPlainText("Server started.");
        m_keepServerRunning = true;
        break;
    case FRMSRVR_START_FAILED:
        ui->toggleServerButton->setText("Start Server");
        ui->statusTextEdit->appendPlainText("Server start failed.");
        m_keepServerRunning = false;
        break;
    case FRMSRVR_CLOSED:
        ui->toggleServerButton->setText("Start Server");
        ui->statusTextEdit->appendPlainText("Server closed.");
        m_keepServerRunning = false;
        break;
    case FRMSRVR_CLOSE_FAILED:
        ui->toggleServerButton->setText("Stop Server");
        ui->statusTextEdit->appendPlainText("Server stop failed.");
        m_keepServerRunning = true;
        break;
    case FRMSRVR_NEW_CONNECTION:
        ui->statusTextEdit->appendPlainText("Incoming connection.");
        break;
    case FRMSRVR_SOCKET_NOT_READABLE:
        ui->statusTextEdit->appendPlainText("Socket not readable.");
        break;
    case FRMSRVR_FRAME_RECEIVED:
        ui->statusTextEdit->appendPlainText("Received frame.");
        break;
    default:
        ui->statusTextEdit->appendPlainText("Unknown worker state.");
    }
}

void FrameServerWidget::on_toggleServerButton_clicked()
{
    if(m_keepServerRunning) {
        emit stopServer(); }
    else {
        emit startServer(); }
}
