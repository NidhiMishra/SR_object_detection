#pragma once
extern void test_yolo(network net, detection_layer l, IplImage im, float thresh);
extern void convert_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);
extern void set_batch_network(network *net, int b);
extern image **load_alphabet();
extern list *read_data_cfg(char *filename);
extern char *option_find_str(list *l, char *key, char *def);
extern char **get_labels(char *filename);
