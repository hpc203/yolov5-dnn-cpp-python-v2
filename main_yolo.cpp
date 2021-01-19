#include "yolo.h"

YOLO::YOLO(Net_config config)
{
	cout << "Net use " << config.netname << endl;
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;
    strcpy(this->netname, config.netname.c_str());

	ifstream ifs(this->classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->classes.push_back(line);

	string modelFile = this->netname;
	modelFile += ".onnx";
	this->net = readNet(modelFile);
}

void YOLO::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	label = this->classes[classId] + ":" + label;

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
}

void YOLO::sigmoid(Mat* out, int length)
{
	float* pdata = (float*)(out->data);
	int i = 0; 
	for (i = 0; i < length; i++)
	{
		pdata[i] = 1.0 / (1 + expf(-pdata[i]));
	}
}

void YOLO::detect(Mat& frame)
{
	Mat blob;
	blobFromImage(frame, blob, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
	
	/////generate proposals
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;
	float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
	int n = 0, q = 0, i = 0, j = 0, nout = this->classes.size() + 5, row_ind = 0;
	for (n = 0; n < 3; n++)   ///�߶�
	{
		int num_grid_x = (int)(this->inpWidth / this->stride[n]);
		int num_grid_y = (int)(this->inpHeight / this->stride[n]);
		for (q = 0; q < 3; q++)    ///anchor��
		{
			const float anchor_w = this->anchors[n][q * 2];
			const float anchor_h = this->anchors[n][q * 2 + 1];
			for (i = 0; i < num_grid_y; i++)
			{
				for (j = 0; j < num_grid_x; j++)
				{
                    float* pdata = (float*)outs[0].data + row_ind * nout;
                    float box_score = sigmoid_x(pdata[4]);
					if (box_score > this->objThreshold)
					{
                        Mat scores = outs[0].row(row_ind).colRange(5, outs[0].cols);
                        Point classIdPoint;
                        double max_class_socre;
                        // Get the value and location of the maximum score
                        minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
                        max_class_socre = sigmoid_x((float)max_class_socre);
						if (max_class_socre > this->confThreshold)
						{
							float cx = (sigmoid_x(pdata[0]) * 2.f - 0.5f + j) * this->stride[n];  ///cx
							float cy = (sigmoid_x(pdata[1]) * 2.f - 0.5f + i) * this->stride[n];   ///cy
							float w = powf(sigmoid_x(pdata[2]) * 2.f, 2.f) * anchor_w;   ///w
							float h = powf(sigmoid_x(pdata[3]) * 2.f, 2.f) * anchor_h;  ///h
							
							int left = (cx - 0.5*w)*ratiow;
							int top = (cy - 0.5*h)*ratioh;   ///���껹ԭ��ԭͼ��

							classIds.push_back(classIdPoint.x);
							confidences.push_back(max_class_socre);
							boxes.push_back(Rect(left, top, (int)(w*ratiow), (int)(h*ratioh)));
						}	
					}
					row_ind++;
				}
			}
		}
	}
	
	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		this->drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

int main()
{
	YOLO yolo_model(yolo_nets[3]);
	string imgpath = "bus.jpg";
	Mat srcimg = imread(imgpath);
	yolo_model.detect(srcimg);
	
	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}