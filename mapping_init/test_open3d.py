#!/usr/bin/env python
"""
created at: Sat Dec  5 00:22:43 2020
created by: Priyam Tejaswin (tejaswin.p@flipkart.com)

Testing Intel's 3D dataviz library.
"""


import open3d as o3d

pcd = o3d.io.read_point_cloud('./test_points.csv', format='xyz')
print(pcd)

o3d.visualization.draw_geometries([pcd])

