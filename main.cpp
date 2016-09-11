#include "rt3dreconst_gui.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    RT3DReconst_GUI w;
    w.show();

    return a.exec();
}
