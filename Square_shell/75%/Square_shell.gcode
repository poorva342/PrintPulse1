;FLAVOR:Marlin
;TIME:386
;Filament used: 0.122397m
;Layer height: 0.2
;MINX:95.2
;MINY:105.2
;MINZ:0.2
;MAXX:104.8
;MAXY:114.8
;MAXZ:10
;TARGET_MACHINE.NAME:Creality Ender-3
;Generated with Cura_SteamEngine 5.7.0
M140 S50
M105
M190 S50
M104 S200
M105
M109 S200
M82 ;absolute extrusion mode
; Ender 3 Custom Start G-code
G92 E0 ; Reset Extruder
G28 ; Home all axes
G1 X12.1 Y20 Z0.3 F5000.0 ; Move to start position
G1 X12.1 Y200.0 Z0.3 F1500.0 E15 ; Draw the first line
G1 X12.4 Y200.0 Z0.3 F5000.0 ; Move to side a little
G1 X12.4 Y20 Z0.3 F1500.0 E30 ; Draw the second line
G92 E0 ; Reset Extruder
G1 Z2.0 F3000 ; Move Z Axis up little to prevent scratching of Heat Bed
G4 S1 ; Wait 1 second
G1 X0 Y0 Z0 F1000.0 ; Move over to prevent blob squish & Go Home
G4 S30 ; Wait 30 seconds
G0 Z0.2 ; Drop to bed
G92 E0
G1 F2700 E-5
;LAYER_COUNT:50
;LAYER:0
M106 S255
;MESH:Square.STL
G0 F6000 X104.4 Y114.4 Z0.2
;TYPE:WALL-INNER
G1 F2700 E0
G1 F600 X104.4 Y105.6 E0.29269
G1 X95.6 Y105.6 E0.58538
G1 X95.6 Y114.4 E0.87807
G1 X104.4 Y114.4 E1.17076
G0 F6000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E1.49005
G1 X95.2 Y105.2 E1.80935
G1 X95.2 Y114.8 E2.12865
G1 X104.8 Y114.8 E2.44795
G0 F6000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z0.4
G0 F6000 X104.4 Y114.4
;TIME_ELAPSED:12.050198
;LAYER:1
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E2.74064
G1 X95.6 Y105.6 E3.03332
G1 X95.6 Y114.4 E3.32601
G1 X104.4 Y114.4 E3.6187
G0 F7500 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E3.938
G1 X95.2 Y105.2 E4.2573
G1 X95.2 Y114.8 E4.57659
G1 X104.8 Y114.8 E4.89589
G0 F7500 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z0.6
G0 F7500 X104.4 Y114.4
;TIME_ELAPSED:19.685386
;LAYER:2
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E5.18858
G1 X95.6 Y105.6 E5.48127
G1 X95.6 Y114.4 E5.77396
G1 X104.4 Y114.4 E6.06665
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E6.38595
G1 X95.2 Y105.2 E6.70524
G1 X95.2 Y114.8 E7.02454
G1 X104.8 Y114.8 E7.34384
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z0.8
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:27.321651
;LAYER:3
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E7.63653
G1 X95.6 Y105.6 E7.92922
G1 X95.6 Y114.4 E8.22191
G1 X104.4 Y114.4 E8.51459
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E8.83389
G1 X95.2 Y105.2 E9.15319
G1 X95.2 Y114.8 E9.47249
G1 X104.8 Y114.8 E9.79178
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z1
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:34.957915
;LAYER:4
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E10.08447
G1 X95.6 Y105.6 E10.37716
G1 X95.6 Y114.4 E10.66985
G1 X104.4 Y114.4 E10.96254
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E11.28184
G1 X95.2 Y105.2 E11.60114
G1 X95.2 Y114.8 E11.92043
G1 X104.8 Y114.8 E12.23973
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z1.2
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:42.594180
;LAYER:5
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E12.53242
G1 X95.6 Y105.6 E12.82511
G1 X95.6 Y114.4 E13.1178
G1 X104.4 Y114.4 E13.41049
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E13.72978
G1 X95.2 Y105.2 E14.04908
G1 X95.2 Y114.8 E14.36838
G1 X104.8 Y114.8 E14.68768
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z1.4
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:50.230445
;LAYER:6
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E14.98036
G1 X95.6 Y105.6 E15.27305
G1 X95.6 Y114.4 E15.56574
G1 X104.4 Y114.4 E15.85843
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E16.17773
G1 X95.2 Y105.2 E16.49703
G1 X95.2 Y114.8 E16.81632
G1 X104.8 Y114.8 E17.13562
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z1.6
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:57.866710
;LAYER:7
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E17.42831
G1 X95.6 Y105.6 E17.721
G1 X95.6 Y114.4 E18.01369
G1 X104.4 Y114.4 E18.30638
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E18.62568
G1 X95.2 Y105.2 E18.94497
G1 X95.2 Y114.8 E19.26427
G1 X104.8 Y114.8 E19.58357
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z1.8
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:65.502974
;LAYER:8
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E19.87626
G1 X95.6 Y105.6 E20.16895
G1 X95.6 Y114.4 E20.46164
G1 X104.4 Y114.4 E20.75432
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E21.07362
G1 X95.2 Y105.2 E21.39292
G1 X95.2 Y114.8 E21.71222
G1 X104.8 Y114.8 E22.03151
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z2
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:73.139239
;LAYER:9
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E22.3242
G1 X95.6 Y105.6 E22.61689
G1 X95.6 Y114.4 E22.90958
G1 X104.4 Y114.4 E23.20227
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E23.52157
G1 X95.2 Y105.2 E23.84086
G1 X95.2 Y114.8 E24.16016
G1 X104.8 Y114.8 E24.47946
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z2.2
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:80.775504
;LAYER:10
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E24.77215
G1 X95.6 Y105.6 E25.06484
G1 X95.6 Y114.4 E25.35753
G1 X104.4 Y114.4 E25.65022
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E25.96951
G1 X95.2 Y105.2 E26.28881
G1 X95.2 Y114.8 E26.60811
G1 X104.8 Y114.8 E26.92741
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z2.4
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:88.411769
;LAYER:11
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E27.22009
G1 X95.6 Y105.6 E27.51278
G1 X95.6 Y114.4 E27.80547
G1 X104.4 Y114.4 E28.09816
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E28.41746
G1 X95.2 Y105.2 E28.73676
G1 X95.2 Y114.8 E29.05605
G1 X104.8 Y114.8 E29.37535
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z2.6
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:96.048033
;LAYER:12
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E29.66804
G1 X95.6 Y105.6 E29.96073
G1 X95.6 Y114.4 E30.25342
G1 X104.4 Y114.4 E30.54611
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E30.86541
G1 X95.2 Y105.2 E31.1847
G1 X95.2 Y114.8 E31.504
G1 X104.8 Y114.8 E31.8233
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z2.8
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:103.684298
;LAYER:13
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E32.11599
G1 X95.6 Y105.6 E32.40868
G1 X95.6 Y114.4 E32.70136
G1 X104.4 Y114.4 E32.99405
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E33.31335
G1 X95.2 Y105.2 E33.63265
G1 X95.2 Y114.8 E33.95195
G1 X104.8 Y114.8 E34.27124
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z3
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:111.320563
;LAYER:14
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E34.56393
G1 X95.6 Y105.6 E34.85662
G1 X95.6 Y114.4 E35.14931
G1 X104.4 Y114.4 E35.442
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E35.7613
G1 X95.2 Y105.2 E36.08059
G1 X95.2 Y114.8 E36.39989
G1 X104.8 Y114.8 E36.71919
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z3.2
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:118.956828
;LAYER:15
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E37.01188
G1 X95.6 Y105.6 E37.30457
G1 X95.6 Y114.4 E37.59726
G1 X104.4 Y114.4 E37.88995
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E38.20924
G1 X95.2 Y105.2 E38.52854
G1 X95.2 Y114.8 E38.84784
G1 X104.8 Y114.8 E39.16714
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z3.4
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:126.593092
;LAYER:16
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E39.45982
G1 X95.6 Y105.6 E39.75251
G1 X95.6 Y114.4 E40.0452
G1 X104.4 Y114.4 E40.33789
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E40.65719
G1 X95.2 Y105.2 E40.97649
G1 X95.2 Y114.8 E41.29578
G1 X104.8 Y114.8 E41.61508
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z3.6
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:134.229357
;LAYER:17
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E41.90777
G1 X95.6 Y105.6 E42.20046
G1 X95.6 Y114.4 E42.49315
G1 X104.4 Y114.4 E42.78584
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E43.10514
G1 X95.2 Y105.2 E43.42443
G1 X95.2 Y114.8 E43.74373
G1 X104.8 Y114.8 E44.06303
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z3.8
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:141.865622
;LAYER:18
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E44.35572
G1 X95.6 Y105.6 E44.64841
G1 X95.6 Y114.4 E44.94109
G1 X104.4 Y114.4 E45.23378
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E45.55308
G1 X95.2 Y105.2 E45.87238
G1 X95.2 Y114.8 E46.19168
G1 X104.8 Y114.8 E46.51097
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z4
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:149.501887
;LAYER:19
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E46.80366
G1 X95.6 Y105.6 E47.09635
G1 X95.6 Y114.4 E47.38904
G1 X104.4 Y114.4 E47.68173
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E48.00103
G1 X95.2 Y105.2 E48.32032
G1 X95.2 Y114.8 E48.63962
G1 X104.8 Y114.8 E48.95892
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z4.2
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:157.138151
;LAYER:20
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E49.25161
G1 X95.6 Y105.6 E49.5443
G1 X95.6 Y114.4 E49.83699
G1 X104.4 Y114.4 E50.12968
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E50.44897
G1 X95.2 Y105.2 E50.76827
G1 X95.2 Y114.8 E51.08757
G1 X104.8 Y114.8 E51.40686
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z4.4
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:164.774416
;LAYER:21
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E51.69955
G1 X95.6 Y105.6 E51.99224
G1 X95.6 Y114.4 E52.28493
G1 X104.4 Y114.4 E52.57762
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E52.89692
G1 X95.2 Y105.2 E53.21622
G1 X95.2 Y114.8 E53.53551
G1 X104.8 Y114.8 E53.85481
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z4.6
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:172.410681
;LAYER:22
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E54.1475
G1 X95.6 Y105.6 E54.44019
G1 X95.6 Y114.4 E54.73288
G1 X104.4 Y114.4 E55.02557
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E55.34486
G1 X95.2 Y105.2 E55.66416
G1 X95.2 Y114.8 E55.98346
G1 X104.8 Y114.8 E56.30276
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z4.8
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:180.046946
;LAYER:23
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E56.59545
G1 X95.6 Y105.6 E56.88813
G1 X95.6 Y114.4 E57.18082
G1 X104.4 Y114.4 E57.47351
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E57.79281
G1 X95.2 Y105.2 E58.11211
G1 X95.2 Y114.8 E58.43141
G1 X104.8 Y114.8 E58.7507
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z5
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:187.683210
;LAYER:24
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E59.04339
G1 X95.6 Y105.6 E59.33608
G1 X95.6 Y114.4 E59.62877
G1 X104.4 Y114.4 E59.92146
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E60.24076
G1 X95.2 Y105.2 E60.56005
G1 X95.2 Y114.8 E60.87935
G1 X104.8 Y114.8 E61.19865
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z5.2
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:195.319475
;LAYER:25
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E61.49134
G1 X95.6 Y105.6 E61.78403
G1 X95.6 Y114.4 E62.07672
G1 X104.4 Y114.4 E62.36941
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E62.6887
G1 X95.2 Y105.2 E63.008
G1 X95.2 Y114.8 E63.3273
G1 X104.8 Y114.8 E63.64659
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z5.4
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:202.955740
;LAYER:26
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E63.93928
G1 X95.6 Y105.6 E64.23197
G1 X95.6 Y114.4 E64.52466
G1 X104.4 Y114.4 E64.81735
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E65.13665
G1 X95.2 Y105.2 E65.45595
G1 X95.2 Y114.8 E65.77524
G1 X104.8 Y114.8 E66.09454
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z5.6
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:210.592005
;LAYER:27
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E66.38723
G1 X95.6 Y105.6 E66.67992
G1 X95.6 Y114.4 E66.97261
G1 X104.4 Y114.4 E67.2653
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E67.58459
G1 X95.2 Y105.2 E67.90389
G1 X95.2 Y114.8 E68.22319
G1 X104.8 Y114.8 E68.54249
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z5.8
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:218.228269
;LAYER:28
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E68.83518
G1 X95.6 Y105.6 E69.12786
G1 X95.6 Y114.4 E69.42055
G1 X104.4 Y114.4 E69.71324
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E70.03254
G1 X95.2 Y105.2 E70.35184
G1 X95.2 Y114.8 E70.67113
G1 X104.8 Y114.8 E70.99043
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z6
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:225.864534
;LAYER:29
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E71.28312
G1 X95.6 Y105.6 E71.57581
G1 X95.6 Y114.4 E71.8685
G1 X104.4 Y114.4 E72.16119
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E72.48049
G1 X95.2 Y105.2 E72.79978
G1 X95.2 Y114.8 E73.11908
G1 X104.8 Y114.8 E73.43838
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z6.2
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:233.500799
;LAYER:30
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E73.73107
G1 X95.6 Y105.6 E74.02376
G1 X95.6 Y114.4 E74.31645
G1 X104.4 Y114.4 E74.60913
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E74.92843
G1 X95.2 Y105.2 E75.24773
G1 X95.2 Y114.8 E75.56703
G1 X104.8 Y114.8 E75.88632
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z6.4
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:241.137064
;LAYER:31
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E76.17901
G1 X95.6 Y105.6 E76.4717
G1 X95.6 Y114.4 E76.76439
G1 X104.4 Y114.4 E77.05708
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E77.37638
G1 X95.2 Y105.2 E77.69568
G1 X95.2 Y114.8 E78.01497
G1 X104.8 Y114.8 E78.33427
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z6.6
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:248.773328
;LAYER:32
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E78.62696
G1 X95.6 Y105.6 E78.91965
G1 X95.6 Y114.4 E79.21234
G1 X104.4 Y114.4 E79.50503
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E79.82432
G1 X95.2 Y105.2 E80.14362
G1 X95.2 Y114.8 E80.46292
G1 X104.8 Y114.8 E80.78222
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z6.8
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:256.409593
;LAYER:33
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E81.07491
G1 X95.6 Y105.6 E81.36759
G1 X95.6 Y114.4 E81.66028
G1 X104.4 Y114.4 E81.95297
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E82.27227
G1 X95.2 Y105.2 E82.59157
G1 X95.2 Y114.8 E82.91086
G1 X104.8 Y114.8 E83.23016
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z7
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:264.045858
;LAYER:34
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E83.52285
G1 X95.6 Y105.6 E83.81554
G1 X95.6 Y114.4 E84.10823
G1 X104.4 Y114.4 E84.40092
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E84.72022
G1 X95.2 Y105.2 E85.03951
G1 X95.2 Y114.8 E85.35881
G1 X104.8 Y114.8 E85.67811
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z7.2
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:271.682123
;LAYER:35
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E85.9708
G1 X95.6 Y105.6 E86.26349
G1 X95.6 Y114.4 E86.55618
G1 X104.4 Y114.4 E86.84886
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E87.16816
G1 X95.2 Y105.2 E87.48746
G1 X95.2 Y114.8 E87.80676
G1 X104.8 Y114.8 E88.12605
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z7.4
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:279.318387
;LAYER:36
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E88.41874
G1 X95.6 Y105.6 E88.71143
G1 X95.6 Y114.4 E89.00412
G1 X104.4 Y114.4 E89.29681
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E89.61611
G1 X95.2 Y105.2 E89.93541
G1 X95.2 Y114.8 E90.2547
G1 X104.8 Y114.8 E90.574
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z7.6
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:286.954652
;LAYER:37
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E90.86669
G1 X95.6 Y105.6 E91.15938
G1 X95.6 Y114.4 E91.45207
G1 X104.4 Y114.4 E91.74476
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E92.06405
G1 X95.2 Y105.2 E92.38335
G1 X95.2 Y114.8 E92.70265
G1 X104.8 Y114.8 E93.02195
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z7.8
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:294.590917
;LAYER:38
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E93.31463
G1 X95.6 Y105.6 E93.60732
G1 X95.6 Y114.4 E93.90001
G1 X104.4 Y114.4 E94.1927
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E94.512
G1 X95.2 Y105.2 E94.8313
G1 X95.2 Y114.8 E95.15059
G1 X104.8 Y114.8 E95.46989
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z8
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:302.227182
;LAYER:39
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E95.76258
G1 X95.6 Y105.6 E96.05527
G1 X95.6 Y114.4 E96.34796
G1 X104.4 Y114.4 E96.64065
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E96.95995
G1 X95.2 Y105.2 E97.27924
G1 X95.2 Y114.8 E97.59854
G1 X104.8 Y114.8 E97.91784
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z8.2
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:309.863446
;LAYER:40
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E98.21053
G1 X95.6 Y105.6 E98.50322
G1 X95.6 Y114.4 E98.79591
G1 X104.4 Y114.4 E99.08859
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E99.40789
G1 X95.2 Y105.2 E99.72719
G1 X95.2 Y114.8 E100.04649
G1 X104.8 Y114.8 E100.36578
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z8.4
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:317.499711
;LAYER:41
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E100.65847
G1 X95.6 Y105.6 E100.95116
G1 X95.6 Y114.4 E101.24385
G1 X104.4 Y114.4 E101.53654
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E101.85584
G1 X95.2 Y105.2 E102.17513
G1 X95.2 Y114.8 E102.49443
G1 X104.8 Y114.8 E102.81373
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z8.6
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:325.135976
;LAYER:42
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E103.10642
G1 X95.6 Y105.6 E103.39911
G1 X95.6 Y114.4 E103.6918
G1 X104.4 Y114.4 E103.98449
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E104.30378
G1 X95.2 Y105.2 E104.62308
G1 X95.2 Y114.8 E104.94238
G1 X104.8 Y114.8 E105.26168
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z8.8
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:332.772241
;LAYER:43
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E105.55436
G1 X95.6 Y105.6 E105.84705
G1 X95.6 Y114.4 E106.13974
G1 X104.4 Y114.4 E106.43243
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E106.75173
G1 X95.2 Y105.2 E107.07103
G1 X95.2 Y114.8 E107.39032
G1 X104.8 Y114.8 E107.70962
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z9
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:340.408505
;LAYER:44
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E108.00231
G1 X95.6 Y105.6 E108.295
G1 X95.6 Y114.4 E108.58769
G1 X104.4 Y114.4 E108.88038
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E109.19968
G1 X95.2 Y105.2 E109.51897
G1 X95.2 Y114.8 E109.83827
G1 X104.8 Y114.8 E110.15757
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z9.2
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:348.044770
;LAYER:45
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E110.45026
G1 X95.6 Y105.6 E110.74295
G1 X95.6 Y114.4 E111.03563
G1 X104.4 Y114.4 E111.32832
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E111.64762
G1 X95.2 Y105.2 E111.96692
G1 X95.2 Y114.8 E112.28622
G1 X104.8 Y114.8 E112.60551
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z9.4
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:355.681035
;LAYER:46
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E112.8982
G1 X95.6 Y105.6 E113.19089
G1 X95.6 Y114.4 E113.48358
G1 X104.4 Y114.4 E113.77627
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E114.09557
G1 X95.2 Y105.2 E114.41486
G1 X95.2 Y114.8 E114.73416
G1 X104.8 Y114.8 E115.05346
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z9.6
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:363.317300
;LAYER:47
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E115.34615
G1 X95.6 Y105.6 E115.63884
G1 X95.6 Y114.4 E115.93153
G1 X104.4 Y114.4 E116.22422
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E116.54351
G1 X95.2 Y105.2 E116.86281
G1 X95.2 Y114.8 E117.18211
G1 X104.8 Y114.8 E117.50141
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z9.8
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:370.953564
;LAYER:48
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E117.79409
G1 X95.6 Y105.6 E118.08678
G1 X95.6 Y114.4 E118.37947
G1 X104.4 Y114.4 E118.67216
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E118.99146
G1 X95.2 Y105.2 E119.31076
G1 X95.2 Y114.8 E119.63005
G1 X104.8 Y114.8 E119.94935
G0 F9000 X103.92 Y114.038
;MESH:NONMESH
G0 F300 X103.92 Y114.038 Z10
G0 F9000 X104.4 Y114.4
;TIME_ELAPSED:378.589829
;LAYER:49
;TYPE:WALL-INNER
;MESH:Square.STL
G1 F600 X104.4 Y105.6 E120.24204
G1 X95.6 Y105.6 E120.53473
G1 X95.6 Y114.4 E120.82742
G1 X104.4 Y114.4 E121.12011
G0 F9000 X104.8 Y114.8
;TYPE:WALL-OUTER
G1 F600 X104.8 Y105.2 E121.43941
G1 X95.2 Y105.2 E121.7587
G1 X95.2 Y114.8 E122.078
G1 X104.8 Y114.8 E122.3973
G0 F9000 X103.92 Y114.038
;TIME_ELAPSED:386.104200
G1 F2700 E117.3973
M140 S0
M107
G91 ;Relative positioning
G1 E-2 F2700 ;Retract a bit
G1 E-2 Z0.2 F2400 ;Retract and raise Z
G1 X5 Y5 F3000 ;Wipe out
G1 Z10 ;Raise Z more
G90 ;Absolute positioning

G1 X0 Y220.0 ;Present print
M106 S0 ;Turn-off fan
M104 S0 ;Turn-off hotend
M140 S0 ;Turn-off bed

M84 X Y E ;Disable all steppers but Z

M82 ;absolute extrusion mode
M104 S0
;End of Gcode
;SETTING_3 {"global_quality": "[general]\\nversion = 4\\nname = Test_Print_Std_Q
;SETTING_3 uality\\ndefinition = creality_ender3\\n\\n[metadata]\\ntype = qualit
;SETTING_3 y_changes\\nquality_type = standard\\nsetting_version = 23\\n\\n[valu
;SETTING_3 es]\\nadhesion_type = none\\nretraction_combing = infill\\nsupport_en
;SETTING_3 able = False\\n\\n", "extruder_quality": ["[general]\\nversion = 4\\n
;SETTING_3 name = Test_Print_Std_Quality\\ndefinition = creality_ender3\\n\\n[me
;SETTING_3 tadata]\\ntype = quality_changes\\nquality_type = standard\\nintent_c
;SETTING_3 ategory = default\\nposition = 0\\nsetting_version = 23\\n\\n[values]
;SETTING_3 \\ncool_fan_full_layer = 3\\ncool_fan_speed = 100\\ncool_min_layer_ti
;SETTING_3 me = 15\\ninfill_sparse_density = 0\\ninitial_bottom_layers = 0\\ntop
;SETTING_3 _layers = 0\\n\\n"]}