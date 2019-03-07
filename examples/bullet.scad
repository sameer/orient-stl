rotate([180,0,0])
union() {
    cylinder(r=10,h=20, center=true);
    translate([0,0,10])
    sphere(r=10);
}
