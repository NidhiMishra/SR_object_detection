#pragma once
extern void test_detector_img(char **names, image **alphabet, network net, image im, float thresh, object *RecObects, int *objectNumPerFrame);
extern void test_detector_img_for_grasping(char **names, image **alphabet, network net, image im, image imFilter, float thresh, object *RecObects, int *objectNumPerFrame);
extern image ipl_to_image(IplImage* src);
extern void object_reminder(image im, char **names, image **alphabet, int classes, object *RecObects, int *objectNumPerFrame);
extern void free_image(image m);
extern void show_image(image p, const char *name);
extern void read_infor_from_txt(object *RecObects, int *objectNumPerFrame);
extern void save_image_jpg(image p, const char *name);
extern void object_show(image im, char **names, image **alphabet, int classes, object *RecObects, int *objectNumPerFrame);
extern void object_show_grasp(image im, char **names, image **alphabet, int classes, object *RecObects, int *objectNumPerFrame);
extern void object_show_person(image im, char **names, image **alphabet, int classes, object *RecObects, int *objectNumPerFrame);
extern void object_show_person_demo_home(image im, char **names, image **alphabet, int classes, object *RecObects, int *objectNumPerFrame, int flag);
extern void objectFilterUsingObjectCategory(object *RecObects, int *objectNumPerFrame, int objEvent);
extern void objectFilterUsingPersonId(object *RecObects, int *objectNumPerFrame, int *personId, int *personCount);
extern void object_vote_mutilframe(object *RecObects, int *objectNumPerFrame);
extern void distanceFilter(object *RecObects, int *objectNumPerFrame, float distanceThreshold);
extern void set_pixel(image m, int x, int y, int c, float val);
extern void draw_even_message_demo_home(int flag);
extern void objectFilterSpecialID(object *RecObects, int *objectNumPerFrame, int personId);
