// Imagine++ project
// Project:  Panorama
// Author:   Pascal Monasse
// Date:     2013/10/08

#include <Imagine/Graphics.h>
#include <Imagine/Images.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <sstream>
using namespace Imagine;
using namespace std;

// Record clicks in two images, until right button click
void getClicks(Window w1, Window w2,
               vector<IntPoint2>& pts1, vector<IntPoint2>& pts2) {
    // ------------- TODO/A completer ----------
    int finish = false;
    IntPoint2 currentPoint;
    int sw = 0;
    Window w;
    int mouse_click;
    while(!finish) {
        mouse_click = anyGetMouse(currentPoint, w,sw);
        if (mouse_click==3){
            finish=true;
            return;
        }
        if (w == w1){
            pts1.push_back(currentPoint);
            setActiveWindow(w1);
            drawCircle(currentPoint,5,Color(55,250,30),3,false);
        } else {
            pts2.push_back(currentPoint);
            setActiveWindow(w2);
            drawCircle(currentPoint,5,Color(250,55,30),3,false);
        }
    }
}

// Return homography compatible with point matches
Matrix<float> getHomography(const vector<IntPoint2>& pts1,
                            const vector<IntPoint2>& pts2) {
    size_t n = min(pts1.size(), pts2.size());
    if(n<4) {
        cout << "Not enough correspondences: " << n << endl;
        return Matrix<float>::Identity(3);
    }
    Matrix<double> A(2*n,8);
    Vector<double> B(2*n);
    // ------------- TODO/A completer ----------
    int k=0;
    for (int i=0; i < 2*n; i+=2){
        double xi = pts1[k][0];
        double yi = pts1[k][1];
        double xip = pts2[k][0];
        double yip = pts2[k][1];

        A(i,0) = xi;
        A(i,1) = yi;
        A(i,2) = 1;
        A(i,3) = 0;
        A(i,4) = 0;
        A(i,5) = 0;
        A(i,6) = -xip*xi;
        A(i,7) = -xip*yi;
        A(i+1,0) = 0;
        A(i+1,1) = 0;
        A(i+1,2) = 0;
        A(i+1,3) = xi;
        A(i+1,4) = yi;
        A(i+1,5) = 1;
        A(i+1,6) = -yip*xi;
        A(i+1,7) = -yip*yi;
        B[i] = xip;
        B[i+1] = yip;
        k++;
    }
    
    B = linSolve(A, B);
    Matrix<float> H(3, 3);
    H(0,0)=B[0]; H(0,1)=B[1]; H(0,2)=B[2];
    H(1,0)=B[3]; H(1,1)=B[4]; H(1,2)=B[5];
    H(2,0)=B[6]; H(2,1)=B[7]; H(2,2)=1;

    // Sanity check
    for(size_t i=0; i<n; i++) {
        float v1[]={(float)pts1[i].x(), (float)pts1[i].y(), 1.0f};
        float v2[]={(float)pts2[i].x(), (float)pts2[i].y(), 1.0f};
        Vector<float> x1(v1,3);
        Vector<float> x2(v2,3);
        x1 = H*x1;
        cout << x1[1]*x2[2]-x1[2]*x2[1] << ' '
             << x1[2]*x2[0]-x1[0]*x2[2] << ' '
             << x1[0]*x2[1]-x1[1]*x2[0] << endl;
    }
    return H;
}
// Grow rectangle of corners (x0,y0) and (x1,y1) to include (x,y)
void growTo(float& x0, float& y0, float& x1, float& y1, float x, float y) {
    if(x<x0) x0=x;
    if(x>x1) x1=x;
    if(y<y0) y0=y;
    if(y>y1) y1=y;    
}

// Panorama construction
void panorama(const Image<Color,2>& I1, const Image<Color,2>& I2,
              Matrix<float> H) {
    Vector<float> v(3);
    float x0=0, y0=0, x1=I2.width(), y1=I2.height();

    v[0]=0; v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=0; v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    cout << "x0 x1 y0 y1=" << x0 << ' ' << x1 << ' ' << y0 << ' ' << y1<<endl;

    Image<Color> I(int(x1-x0), int(y1-y0));
    setActiveWindow( openWindow(I.width(), I.height()) );
    I.fill(WHITE);

    // ------------- TODO/A completer ----------
    Matrix<float> H_1 = inverse(H);
    Vector<float> v_p(3);
    int x0p;
    int y0p;

    // start from the projected coordinate
    for (int i=int(x0); i < int(x1); i++){
        for (int j=int(y0); j < int(y1); j++){
            x0p = int(x0);
            y0p = int(y0);

            // value to add with i and j for filling the new image I
            x0p = int(x0p) * -1;
            y0p = int(y0p) * -1;
            
            // inverse projection (inverse warping) of (i,j) to get associated value in I1 -> projecting I1 in the view of I2 
            v_p[0]=i; v_p[1]=j; v_p[2]=1;
            v_p=H_1*v_p; v_p/=v_p[2];
            if (i >=0 && i < I2.size(0) && j >=0 && j < I2.size(1)){
                // To enable overlap averaging decomment following part
                /*if (v_p[0] > 0 && v_p[0] < I1.size(0) && v_p[1] > 0 && v_p[1] < I1.size(1)){
                    I(i+x0p,j+y0p)[0] = (I2(i,j)[0] + I1.interpolate(v_p[0], v_p[1])[0] ) /2;
                    I(i+x0p,j+y0p)[1] = (I2(i,j)[0] + I1.interpolate(v_p[0], v_p[1])[1] ) /2;
                    I(i+x0p,j+y0p)[2] = (I2(i,j)[0] + I1.interpolate(v_p[0], v_p[1])[2] ) /2;
                    
                    // without overlap averaging
                    I(i+x0p,j+y0p) = I2(i,j);
                } else {*/
                I(i+x0p,j+y0p) = I2(i,j);
                //}
            } else{
                if (v_p[0] > 0 && v_p[0] < I1.size(0) && v_p[1] > 0 && v_p[1] < I1.size(1)){
                    // interpolation to get corresponding value un I1
                    I(i+x0p,j+y0p)= I1.interpolate(v_p[0], v_p[1]);
                }
            }
        }
    }
    display(I,0,0);
}

// Main function
int main(int argc, char* argv[]) {
    
    const char* s1 = argc>1? argv[1]: srcPath("image0006.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("image0007.jpg");
    
    /*const char* s1 = argc>1? argv[1]: srcPath("image0001.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("image0002.jpg");*/

    // Load and display images
    Image<Color> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load the images" << endl;
        return 1;
    }
    Window w1 = openWindow(I1.width(), I1.height(), s1);
    display(I1,0,0);
    Window w2 = openWindow(I2.width(), I2.height(), s2);
    setActiveWindow(w2);
    display(I2,0,0);

    // Get user's clicks in images
    vector<IntPoint2> pts1, pts2;
    getClicks(w1, w2, pts1, pts2);

    vector<IntPoint2>::const_iterator it;
    cout << "pts1="<<endl;
    for(it=pts1.begin(); it != pts1.end(); it++)
        cout << *it << endl;
    cout << "pts2="<<endl;
    for(it=pts2.begin(); it != pts2.end(); it++)
        cout << *it << endl;

    // Compute homography
    Matrix<float> H = getHomography(pts1, pts2);
    cout << "H=" << H/H(2,2);

    // Apply homography
    panorama(I1, I2, H);

    endGraphics();
    return 0;
}
