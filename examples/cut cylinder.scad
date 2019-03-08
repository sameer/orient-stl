rotate([-150,0,0])
difference() {
    cylinder(r=10,h=20, center=true);
    translate([0,0,10])
    rotate([60,0,0])
    cube([20,20,40], center=true);
}
