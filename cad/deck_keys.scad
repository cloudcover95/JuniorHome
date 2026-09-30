rows=8; cols=8; pitch=19.05; cut=14;
difference(){
  cube([cols*pitch+6, rows*pitch+6, 1.5]);
  for(i=[0:cols-1]) for(j=[0:rows-1])
    translate([3+i*pitch+(pitch-cut)/2, 3+j*pitch+(pitch-cut)/2, -1]) cube([cut,cut,4]);
}
