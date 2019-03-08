// Reuleaux Tetrahedron
// ceptimus 2018-01-13

width = 1; // width of the object to be printed
$fn = 50; // higher numbers give smoother object but (much) longer render times

// vertices and edge length of Reuleaux Tetrahedron from Wikipedia
v1 = [sqrt(8/9), 0, -1/3];
v2 = [-sqrt(2/9), sqrt(2/3), -1/3];
v3 = [-sqrt(2/9), -sqrt(2/3), -1/3];
v4 = [0, 0, 1];
a = sqrt(8/3); // edge length given by above vertices

k = width / a; // scaling constant to generate desired width from standard vertices

reuleauxTetrahedron();

module reuleauxTetrahedron() {
    intersection() {
        translate(v1 * k)
            sphere(r = width);
        translate(v2 * k)
            sphere(r = width);
        translate(v3 * k)
            sphere(r = width);
        translate(v4 * k)
            rotate([90, 0, 0]) // use similar part of sphere for bottom face as other faces - OpenScad renders the parts of 'spheres' near the 'poles' differently
                sphere(r = width);
    }
}
