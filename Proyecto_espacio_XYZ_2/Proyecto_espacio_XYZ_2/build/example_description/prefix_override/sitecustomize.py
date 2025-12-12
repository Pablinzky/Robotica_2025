import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/robousr/Semestre-2026-1/Workspaces/Proyectos_Lalo/Proyecto_Robótica/Proyecto_espacio_XYZ_2/install/example_description'
