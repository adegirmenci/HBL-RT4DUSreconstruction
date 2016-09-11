#ifndef FRAMESERVERWIDGET_H
#define FRAMESERVERWIDGET_H

#include <QWidget>

#include "frameserverthread.h"

namespace Ui {
class FrameServerWidget;
}

class FrameServerWidget : public QWidget
{
    Q_OBJECT

public:
    explicit FrameServerWidget(QWidget *parent = 0);
    ~FrameServerWidget();

    FrameServerThread *m_worker;

signals:
    void startServer();
    void stopServer();

private slots:
    void workerStatusChanged(int status);

    void on_toggleServerButton_clicked();

private:
    Ui::FrameServerWidget *ui;

    bool m_keepServerRunning;

    QThread m_thread; // FrameServerThread will live in here
};

#endif // FRAMESERVERWIDGET_H
