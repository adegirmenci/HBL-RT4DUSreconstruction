#include "rt3dreconst_gui.h"
#include "ui_rt3dreconst_gui.h"

RT3DReconst_GUI::RT3DReconst_GUI(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::RT3DReconst_GUI)
{
    ui->setupUi(this);

    m_worker = new RT3DReconstWorker;
    m_worker->moveToThread(&m_thread);

    connect(&m_thread, SIGNAL(finished()), m_worker, SLOT(deleteLater()));
    m_thread.start();

    connect(m_worker, SIGNAL(statusChanged(int)), this, SLOT(workerStatusChanged(int)));
    connect(m_worker, SIGNAL(volumeSaved(QString)),
            ui->volumeClientWidget, SLOT(sendVolume(QString)));

    connect(ui->frameServerWidget->m_worker, SIGNAL(newFrameReceived(FrameExtd)),
            m_worker, SLOT(addFrame(FrameExtd)));
    connect(ui->volumeClientWidget, SIGNAL(tellVolumeToSave()),
            m_worker, SLOT(tellVolumeToSave()));

    m_worker->initializeWorker();
}

RT3DReconst_GUI::~RT3DReconst_GUI()
{
    m_thread.quit();
    m_thread.wait();
    qDebug() << "RT3DReconst_GUI thread quit.";

    delete ui;
}

void RT3DReconst_GUI::workerStatusChanged(int status)
{
//    switch(status)
//    {
//    case FRMSRVR_STARTED:
//        ui->toggleServerButton->setText("Stop Server");
//        ui->addrPortLineEdit->setText(tr("%1:%2")
//                                      .arg(m_worker->getServerAddress().toString())
//                                      .arg(m_worker->getServerPort()));
//        ui->statusTextEdit->appendPlainText("Server started.");
//        m_keepServerRunning = true;
//        break;
//    case FRMSRVR_START_FAILED:
//        ui->toggleServerButton->setText("Start Server");
//        ui->statusTextEdit->appendPlainText("Server start failed.");
//        m_keepServerRunning = false;
//        break;
//    case FRMSRVR_CLOSED:
//        ui->toggleServerButton->setText("Start Server");
//        ui->statusTextEdit->appendPlainText("Server closed.");
//        m_keepServerRunning = false;
//        break;
//    case FRMSRVR_CLOSE_FAILED:
//        ui->toggleServerButton->setText("Stop Server");
//        ui->statusTextEdit->appendPlainText("Server stop failed.");
//        m_keepServerRunning = true;
//        break;
//    case FRMSRVR_NEW_CONNECTION:
//        ui->statusTextEdit->appendPlainText("Incoming connection.");
//        break;
//    case FRMSRVR_SOCKET_NOT_READABLE:
//        ui->statusTextEdit->appendPlainText("Socket not readable.");
//        break;
//    case FRMSRVR_FRAME_RECEIVED:
//        ui->statusTextEdit->appendPlainText("Received frame.");
//        break;
//    default:
//        ui->statusTextEdit->appendPlainText("Unknown worker state.");
//    }
}

void RT3DReconst_GUI::on_interpButton_clicked()
{
    m_worker->interpolate();
}
