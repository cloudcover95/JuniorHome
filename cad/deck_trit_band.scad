// Clip-on trit / gamma-drift band. Slides into deck_shell left slot.
// Print standing on the 6 mm face. Channel is a diffuser, not an MCU.
$fn=24;
L=104+158.4+137+12; // ~411 along shell height stack, trim to fit
L=280;
thk=6; w=12; slot=2; lip=1.6;
difference(){
  cube([w,L,thk]);
  // gamma channel (LED strip or printed trit ticket)
  translate([2,4,thk-2.2]) cube([w-4, L-8, 2.4]);
  // snap tongue → shell 2.2 mm slot
  translate([w-1.8,-0.1,1]) cube([2, L+0.2, 3.8]);
}
// three detent nubs
for(y=[20, L/2, L-20])
  translate([w-0.6,y,2.2]) cube([1.2,6,1.6]);
