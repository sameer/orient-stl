rotate([-90,0])
translate([0,-10,0])
rotate([90,0,0])
difference() {
    cylinder(r=10,h=20, center=true);
    translate([0,0,6])
    rotate([60,0,0])
    cube([20,20,40], center=true);
}
