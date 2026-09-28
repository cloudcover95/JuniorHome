pitch=24; dia=18; rows=4; cols=4;
difference(){
  cube([cols*pitch+8, rows*pitch+8, 8]);
  for(i=[0:cols-1]) for(j=[0:rows-1])
    translate([8+i*pitch, 8+j*pitch, 3]) cylinder(h=6, d=dia, $fn=24);
}
