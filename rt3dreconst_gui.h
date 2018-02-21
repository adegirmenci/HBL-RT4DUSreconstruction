#ifndef RT3DRECONST_GUI_H
#define RT3DRECONST_GUI_H

#include <QMainWindow>
#include "rt3dreconstworker.h"

namespace Ui {
class RT3DReconst_GUI;
}

class RT3DReconst_GUI : public QMainWindow
{
    Q_OBJECT

public:
    explicit RT3DReconst_GUI(QWidget *parent = 0);
    ~RT3DReconst_GUI();

private slots:
    void workerStatusChanged(int status);

    void on_interpButton_clicked();

    void on_loadDataButton_clicked();

private:
    Ui::RT3DReconst_GUI *ui;

    RT3DReconstWorker *m_worker;

    bool m_keepWorkerRunning;

    QThread m_thread; // FrameServerThread will live in here
};

#endif // RT3DRECONST_GUI_H
