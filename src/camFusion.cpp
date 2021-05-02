
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    BoundingBox bb;
    double meanEucDist {0.0};
    double distTotal {0.0};

    // fill the keypoints and keypoints matches in a temporary boundingBox data structure 
    for (cv::DMatch match: kptMatches)
    {        
        //check if the bounding boxe has the keypoint match
        if (boundingBox.roi.contains(kptsPrev[match.queryIdx].pt) && boundingBox.roi.contains(kptsCurr[match.trainIdx].pt))
        {
            //accumelate euclidean distances between two matched keypoints
            distTotal += sqrt(pow(kptsCurr[match.trainIdx].pt.x - kptsPrev[match.queryIdx].pt.x, 2) + 
                            pow(kptsCurr[match.trainIdx].pt.y - kptsPrev[match.queryIdx].pt.y, 2) * 1.0);

            bb.kptMatches.push_back(match); // increase keypoint counter
            bb.keypoints.push_back(kptsPrev[match.queryIdx]);
        }
    }

    meanEucDist = distTotal/kptMatches.size();

    // filtering outliers using euclidean mean and filling the boundingBox data structure 
    for (cv::DMatch match: bb.kptMatches)
    {
        //accumelate euclidean distances between two matched keypoints
        int dist = sqrt(pow(kptsCurr[match.trainIdx].pt.x - kptsPrev[match.queryIdx].pt.x, 2) + 
                        pow(kptsCurr[match.trainIdx].pt.y - kptsPrev[match.queryIdx].pt.y, 2) * 1.0);
        
        //check if the bounding boxe has the keypoint match
        if (abs(meanEucDist - dist) > 1)
        {
            boundingBox.kptMatches.push_back(match); // increase keypoint counter
            boundingBox.keypoints.push_back(kptsPrev[match.queryIdx]);
        }
    }    

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    float disThreshhold {0.05};
    filterLidarPoints(lidarPointsPrev,disThreshhold);
    filterLidarPoints(lidarPointsCurr,disThreshhold);

    double dT = 1/frameRate; // time between two measurements in seconds
    double laneWidth = 4.0;  // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        if (abs(it->y) <= laneWidth / 2.0)
        {
            minXPrev = minXPrev > it->x ? it->x : minXPrev;
        }
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        if (abs(it->y) <= laneWidth / 2.0)
        {
            minXCurr = minXCurr > it->x ? it->x : minXCurr;
        }    
    }
    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
}



// filters lidar points from outliers using euclidean clustering algorithm
void filterLidarPoints (std::vector<LidarPoint> &lidarPoints, float disThreshhold)
{
    KdTree* tree = new KdTree; //instantiate tree object
    //fill the tree with points
    std::vector<std::vector<float>> points {0};
    for (int i=0; i<lidarPoints.size(); i++)
    {
        std::vector<float> point {lidarPoints[i].x,lidarPoints[i].y};
        points.push_back(point);
        tree->insert(point,i);
    }
    std::vector<std::vector<int>> clusters = euclideanCluster(points, tree, disThreshhold);
    //cout << "number of clusters: " << clusters.size() << endl;

    int largeClusterIndex {-1};
    int clusterSizeCompare {0};
    for (int i = 0; i < clusters.size(); i++)
    {
        if(clusters[i].size() > clusterSizeCompare)
        {
            clusterSizeCompare = clusters[i].size();
            largeClusterIndex = i;
        }
        
    }
    //cout << "size of larger cluster" << "is: " << clusters[largeClusterIndex].size() <<endl;

    std::vector<LidarPoint> cluster;
    for (int index : clusters[largeClusterIndex])
    {
        LidarPoint l;
        l.r = lidarPoints[index].r;
        l.x = lidarPoints[index].x;
        l.r = lidarPoints[index].y;
        l.z = lidarPoints[index].z;
        cluster.push_back(l);
    }
    lidarPoints = cluster;
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    //iterate overall bounding boxes from previous frame
    for( BoundingBox prevFbb : prevFrame.boundingBoxes)
    {   
        int kpNum = 0;
        int prevKpNum = 0;
        int matchID = -1;

        //iterate overall bounding boxes from current frame
        for ( BoundingBox currFbb : currFrame.boundingBoxes)
        {   
            kpNum = 0;
            // itertating over all matches
            for (cv::DMatch match: matches)
            {
                //check if the two bounding boxes has the samme keypoint
                if (prevFbb.roi.contains(prevFrame.keypoints[match.queryIdx].pt) && currFbb.roi.contains(currFrame.keypoints[match.trainIdx].pt))
                {
                    kpNum +=1; // increase keypoint counter
                }
            }
            //check if this current frame bounding box has higher number of correspondence
            if (kpNum > prevKpNum)
            {
                matchID = currFbb.boxID;
                prevKpNum = kpNum;
            }
            
        }// end of the inner loop


        bbBestMatches.insert(pair<int, int>(prevFbb.boxID,matchID));// inster the match with highest number of keypoints correspondence
    }// end of the outer loop
}

void clusterHelper(int indice, const std::vector<std::vector<float>> points, std::vector<int>& cluster, std::vector<bool>& processed, KdTree* tree, float distanceTol)
{
	processed[indice] = true;
	cluster.push_back(indice);

	std::vector<int> nearest = tree->search(points[indice],distanceTol);

	for(int id: nearest)
	{
		if(!processed[id])
			clusterHelper(id, points, cluster, processed, tree, distanceTol);
	}
}

std::vector<std::vector<int>> euclideanCluster(const std::vector<std::vector<float>>& points, KdTree* tree, float distanceTol)
{

	std::vector<std::vector<int>> clusters;

	std::vector<bool> processed(points.size(),false);

	int i = 0;
	while(i < points.size())
	{
		if(processed[i])
		{
			i ++;
			continue;
		}

		std::vector<int> cluster;
		clusterHelper(i, points, cluster, processed, tree, distanceTol);
		clusters.push_back(cluster);
	}
 
	return clusters;
}

