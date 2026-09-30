// JuniorDeck production shell v0.2 — sits on deck_well + deck_keys.
// Print flat, 0.2 mm, no support. trit_mcu false.
$fn=32;
pad_p=24; pad_d=18; pad_n=4;
well=pad_n*pad_p+8; // 104
key_p=19.05; key_n=8; key_cut=14;
keys=key_n*key_p+6; // 158.4
disp_w=218; disp_h=137;
wall=3; lip=2; h=16;
W=disp_w+wall*2+28; // knob column
H=well+keys+disp_h+wall*4;

module geode_ring(x,y){
  translate([x,y,h-1.2]) difference(){
    cylinder(h=1.2,d=pad_d+6);
    translate([0,0,-0.1]) cylinder(h=1.6,d=pad_d+1);
  }
}

difference(){
  cube([W,H,h]);
  // well pocket
  translate([wall,wall,h-9]) cube([well+0.4,well+0.4,10]);
  // pad through-holes + flow rings sit on top
  for(i=[0:pad_n-1]) for(j=[0:pad_n-1])
    translate([wall+8+i*pad_p, wall+8+j*pad_p, -1])
      cylinder(h=h+2,d=pad_d+0.6);
  // key pocket
  translate([wall, wall+well+wall, h-3]) cube([keys+0.4,keys+0.4,4]);
  for(i=[0:key_n-1]) for(j=[0:key_n-1])
    translate([wall+3+i*key_p+(key_p-key_cut)/2,
               wall+well+wall+3+j*key_p+(key_p-key_cut)/2, -1])
      cube([key_cut,key_cut,h+2]);
  // display window
  translate([wall, wall+well+wall+keys+wall, 2])
    cube([disp_w, disp_h, h]);
  // knobs
  for(k=[0:3])
    translate([W-14, wall+20+k*30, -1]) cylinder(h=h+2,d=8);
  // trit-band slot (left edge)
  translate([-0.1, wall, 4]) cube([2.2, H-wall*2, 6]);
}
// geode flow rings around pads (print in place, 1.2 mm)
for(i=[0:pad_n-1]) for(j=[0:pad_n-1])
  geode_ring(wall+8+i*pad_p, wall+8+j*pad_p);
