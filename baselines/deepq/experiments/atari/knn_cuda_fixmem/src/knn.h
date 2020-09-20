struct KNNAddress{
  float  *query_dev;
  float  *ref_dev;
  float  *dist_dev;
  int    *ind_dev;
  bool * musk_dev;

};
extern struct KNNAddress theAddress[20];
extern int address_count;
void allocate_cuda(int address_num,int size, int dims, int query_max, int k);
void add_cuda(int address_num,int place,float* ref_vector, int dims);
void knn_cuda(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host, int* ind_host);
void knn_cuda_fix_ref(int address_num,float* dist_host, int* ind_host, float* query_host, int query_width, int k, int ref_width, int dims);
void knn_cuda_fix_ref_conditional(int address_num,float* dist_host, int* ind_host, float* query_host, bool* musk, int query_width, int k, int ref_width, int dims);
void extract_cuda(float* activation, int n_batch, int n_channel, int height,
    int width, float* coords, int n_max_coord, int dim_coord, float* extracted_activation);
