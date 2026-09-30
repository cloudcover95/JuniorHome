// Entire top is a geode field. Trit-flow grooves run between rings.
// Additive to deck_shell.scad. Print flat 0.2 mm, no support.
$fn=28;
pad_p=24; pad_d=18; n=4; wall=3; h=16;
well=n*pad_p+8;
key_p=19.05; kn=8; kcut=14; keys=kn*key_p+6;
W=218+wall*2+28; H=well+keys+137+wall*4;

module ring(x,y,od){
  translate([x,y,h-1.0]) difference(){
    cylinder(h=1.0,d=od);
    translate([0,0,-0.2]) cylinder(h=1.4,d=od-3.2);
  }
}
module flow(x1,y1,x2,y2){
  hull(){
    translate([x1,y1,h-0.7]) cube([1.6,1.6,0.7],center=true);
    translate([x2,y2,h-0.7]) cube([1.6,1.6,0.7],center=true);
  }
}

difference(){
  cube([W,H,h]);
  translate([wall,wall,h-9]) cube([well+0.4,well+0.4,10]);
  for(i=[0:n-1]) for(j=[0:n-1])
    translate([wall+8+i*pad_p,wall+8+j*pad_p,-1]) cylinder(h=h+2,d=pad_d+0.6);
  translate([wall,wall+well+wall,h-3]) cube([keys+0.4,keys+0.4,4]);
  for(i=[0:kn-1]) for(j=[0:kn-1])
    translate([wall+3+i*key_p+(key_p-kcut)/2,
               wall+well+wall+3+j*key_p+(key_p-kcut)/2,-1]) cube([kcut,kcut,h+2]);
  translate([wall,wall+well+wall+keys+wall,2]) cube([218,137,h]);
  for(k=[0:3]) translate([W-14,wall+20+k*30,-1]) cylinder(h=h+2,d=8);
  translate([-0.1,wall,4]) cube([2.2,H-wall*2,6]);
}
// geode field on pad face
for(i=[0:n-1]) for(j=[0:n-1]){
  cx=wall+8+i*pad_p; cy=wall+8+j*pad_p;
  ring(cx,cy,pad_d+6);
  ring(cx,cy,pad_d+10);
  if(i<n-1) flow(cx,cy,cx+pad_p,cy);
  if(j<n-1) flow(cx,cy,cx,cy+pad_p);
}
