// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <limits>

using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match {
    float x1, y1, x2, y2;
};

// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}

float dist_epipolar(Match m, FMatrix<float, 3,3> F){
    FVector<float,3> x;
    x[0] = m.x1;
    x[1] = m.y1;
    x[2] = 1;
    FMatrix<float,3,3> Ft = transpose(F);
    float num = abs(((Ft*x)[0] * m.x2) + ((Ft*x)[1] * m.y2) + (Ft*x)[2]);
    float denum = sqrt(pow((Ft*x)[0],2) + pow((Ft*x)[1],2));
    return num/denum;
}

FMatrix<float,3,3> Fcomputation1(vector<Match> best_m, FMatrix<float,3,3> NORM){

    FMatrix<float,9,9> A;
    for (int i=0; i < 8; i++){
        float xi = best_m[i].x1 * 0.001;
        float xip = best_m[i].x2* 0.001;
        float yi = best_m[i].y1* 0.001;
        float yip = best_m[i].y2* 0.001;
        A(i,0) = xi*xip; // F11
        A(i,1) = xi*yip; // F12
        A(i,2) = xi; // F13
        A(i,3) = yi*xip; // F21
        A(i,4) = yi*yip; // F22
        A(i,5) = yi; // F23
        A(i,6) = xip; // F31
        A(i,7) = yip; // F32
        A(i,8) = 1; // F33
    }
    A(8,0) = 0; A(8,1) = 0; A(8,2) = 0; A(8,3) = 0; A(8,4) = 0; A(8,5) = 0; A(8,6) = 0; A(8,7) = 0; A(8,8) = 0;

    FMatrix<float,9,9> U;
    FVector<float,9> S(9);
    FMatrix<float,9,9> Vt;

    svd(A,U,S,Vt);

    FMatrix<float, 3,3> F1;

    F1(0,0) = Vt.getRow(8)[0];
    F1(0,1) = Vt.getRow(8)[1];
    F1(0,2) = Vt.getRow(8)[2];
    F1(1,0) = Vt.getRow(8)[3];
    F1(1,1) = Vt.getRow(8)[4];
    F1(1,2) = Vt.getRow(8)[5];
    F1(2,0) = Vt.getRow(8)[6];
    F1(2,1) = Vt.getRow(8)[7];
    F1(2,2) = Vt.getRow(8)[8];

    FMatrix<float, 3,3> U1;
    FVector<float, 3> S1;
    FMatrix<float, 3,3> Vt1;
    svd(F1,U1,S1,Vt1);

    float sigma[3][3] =
    {
        {S1[0],    0,      0},                   // force s3 = 0
        {0,         S1[1], 0},
        {0,         0,      0}
    };
    FMatrix<float,3,3> sigma_M(sigma);

    F1 = U1 * sigma_M * Vt1;

    FMatrix<float,3,3> F;
    F = transpose(NORM) * F1 * NORM;

    return F;
}

FMatrix<float,3,3> Fcomputation2(vector<Match> best_m, FMatrix<float,3,3> NORM){

    Matrix<float> A(best_m.size(),9);
    for (int i=0; i < (int)best_m.size(); i++){
        float xi = best_m[i].x1 *0.001;
        float xip = best_m[i].x2*0.001;
        float yi = best_m[i].y1*0.001;
        float yip = best_m[i].y2*0.001;
        A(i,0) = xi*xip; // F11
        A(i,1) = xi*yip; // F12
        A(i,2) = xi; // F13
        A(i,3) = yi*xip; // F21
        A(i,4) = yi*yip; // F22
        A(i,5) = yi; // F23
        A(i,6) = xip; // F31
        A(i,7) = yip; // F32
        A(i,8) = 1; // F33
    }

    Matrix<float> U(best_m.size(),9);
    Vector<float> S(9);
    Matrix<float> Vt(best_m.size(),9);

    svd(A,U,S,Vt);

    FMatrix<float, 3,3> F1;

    F1(0,0) = Vt.getRow(8)[0];
    F1(0,1) = Vt.getRow(8)[1];
    F1(0,2) = Vt.getRow(8)[2];
    F1(1,0) = Vt.getRow(8)[3];
    F1(1,1) = Vt.getRow(8)[4];
    F1(1,2) = Vt.getRow(8)[5];
    F1(2,0) = Vt.getRow(8)[6];
    F1(2,1) = Vt.getRow(8)[7];
    F1(2,2) = Vt.getRow(8)[8];

    FMatrix<float, 3,3> U1;
    FVector<float, 3> S1;
    FMatrix<float, 3,3> Vt1;
    svd(F1,U1,S1,Vt1);

    float sigma[3][3] =
    {
        {S1[0],    0,      0},                   // force s3 = 0
        {0,         S1[1], 0},
        {0,         0,      0}
    };
    FMatrix<float,3,3> sigma_M(sigma);

    F1 = U1 * sigma_M * Vt1;

    FMatrix<float,3,3> F;
    F = transpose(NORM) * F1 * NORM;

    return F;
}

// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float,3,3> computeF(vector<Match>& matches) {
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination // T
    int Niter=100000; // Adjusted dynamically

    vector<int> bestInliers;

    vector<int> bestF_index;
    int n_iter = 0;
    vector<Match> best_m;
    vector<int> ens_pts_index;
    vector<Match> ens_pts_m;
    FMatrix<float,3,3> bestF;
    int k=8;

    float norm[3][3] =
    {
        {0.001, 0,      0},
        {0,     0.001,  0},
        {0,     0,      1}
    };
    FMatrix<float,3,3> N(norm);

    while(n_iter < Niter){
        ens_pts_m.clear();
        ens_pts_index.clear();

        // k random points selection
        for(int i=0; i<k; i++){
            int r = std::rand() % (int)matches.size();
            ens_pts_m.push_back(matches[r]);
            ens_pts_index.push_back(r);
        }

        // model F - FIT
        FMatrix<float,3,3> F = Fcomputation1(ens_pts_m, N);

        // Test model F
        cout << "n_iter " << n_iter << " " << Niter << endl;
        for(int i=0; i < (int)matches.size(); i++){
            if (!std::count(ens_pts_index.begin(), ens_pts_index.end(), i)){
                // if the point ajust to model with error less than distMax
                float dist = dist_epipolar(matches[i], F);

                if (dist <= distMax){
                    ens_pts_m.push_back(matches[i]);
                    ens_pts_index.push_back(i);
                }
            }
        }

        // update model
        if((int)ens_pts_m.size() > (int)best_m.size()){
            best_m.swap(ens_pts_m);
            bestInliers.swap(ens_pts_index);

            Niter = ceil( log(BETA) / log(1.0-pow((float)bestInliers.size()/matches.size(),k)) );
            bestF = Fcomputation2(best_m, N); // recompute model on new points

        }
        n_iter++;
    }

    // Updating matches with inliers only
    vector<Match> all=matches;
    matches.clear();
    for(size_t i=0; i<bestInliers.size(); i++)
        matches.push_back(all[bestInliers[i]]);
    return bestF;
}

// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2,
                     const FMatrix<float,3,3>& F) {

    Color color(255,0,0);

    while(true) {
        int x,y;
        if(getMouse(x,y) == 3)
            break;

        // Fx' = epipolar line in left image associated with x'
        // Ftx = epipolar line in right image associated with x
        fillCircle(x, y, 3, color);
        FVector<float,3> line;
        FVector<float,3> pts;
        pts[0] = x;
        pts[1] = y;
        pts[2] = 1;
        bool img1 = false;

        if (x <= I1.width()){ // Click in image 1
            line = transpose(F)*pts;
            img1 = true;
        } else{ // Click in image 2
            pts[0] -= I1.width();
            line = F*pts;
        }

        if (img1){
            drawLine(I1.width(),(-1*line[2])/line[1],2*I1.width(),-1*(line[2]+line[0]*I1.width())/line[1],color,2);
        } else{
            drawLine(0,(-1*line[2])/line[1],I1.width(),-1*(line[2]+line[0]*I1.width())/line[1],color,2);
        }
    }
}

int main(int argc, char* argv[])
{
    srand((unsigned int)time(0));

    const char* s1 = argc>1? argv[1]: srcPath("im1.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("im2.jpg");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    const int n = (int)matches.size();
    cout << " matches: " << n << endl;
    drawString(100,20,std::to_string(n)+ " matches",RED);
    click();
    
    FMatrix<float,3,3> F = computeF(matches);
    cout << "F="<< endl << F;

    // Redisplay with matches
    display(I1,0,0);
    display(I2,w,0);
    for(size_t i=0; i<matches.size(); i++) {
        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(matches[i].x1+0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2+w, matches[i].y2, 2, c);        
    }
    drawString(100, 20, to_string(matches.size())+"/"+to_string(n)+" inliers", RED);
    click();

    // Redisplay without SIFT points
    display(I1,0,0);
    display(I2,w,0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}
