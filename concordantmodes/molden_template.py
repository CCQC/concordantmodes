molden_template = """[Molden Format]                                                                 
[ATOMS] AU                                                                     
{geometry}
[Molden Format]                                                                
 [FREQ]                                                                         
{frequencies}
 [FR-COORD] 
{fr_geom}
 [FR-NORM-COORD]                                                               
{normal_modes}"""
