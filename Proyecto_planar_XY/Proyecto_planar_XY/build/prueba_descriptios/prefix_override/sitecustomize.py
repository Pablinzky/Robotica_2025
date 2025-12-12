import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/robousr/Semestre-2026-1/Workspaces/Proyectos_Lalo/Proyecto_Robótica/example_ws_pkg_URDF_renamed_colors2/install/prueba_descriptios'
