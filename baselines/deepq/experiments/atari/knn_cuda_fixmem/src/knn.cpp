// Python
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
#include "knn.h"
#include <cstdio>
using namespace boost::python;

int address_count =0;
struct KNNAddress theAddress[20]={{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0}};
// For extracting features from a 4-D blob feature map.
object extract_feature(PyObject* activation_, PyObject* coords_)
{
  PyArrayObject* activation_py = (PyArrayObject*) activation_;
  PyArrayObject* coords_py     = (PyArrayObject*) coords_;
  int n_batch   = activation_py->dimensions[0];
  int n_channel = activation_py->dimensions[1];
  int height    = activation_py->dimensions[2];
  int width     = activation_py->dimensions[3];

  int n_max_coord = coords_py->dimensions[1];
  int dim_coord   = coords_py->dimensions[2];

  float* activation           = new float[n_batch * n_channel * height * width];
  float* coords               = new float[n_batch * n_max_coord * dim_coord];
  float* extracted_activation = new float[n_batch * n_channel * n_max_coord];;

  // Copy python objects
  for(int n = 0; n < n_batch; n++){
    for (int c = 0; c < n_channel; c++){
      for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
          activation[((n * n_channel + c) * height + i) * width + j] =
              *(float*)PyArray_GETPTR4(activation_py, n, c, i, j);
        }
      }
    }
  }

  for(int n = 0; n < n_batch; n++){
    for(int i = 0; i < n_max_coord; i++) {
      for(int j = 0; j < dim_coord; j++) {
        coords[(n * n_max_coord + i) * dim_coord + j] =
            *(float*)PyArray_GETPTR3(coords_py, n, i, j);
      }
    }
  }

  extract_cuda(activation, n_batch, n_channel, height,
      width, coords, n_max_coord, dim_coord, extracted_activation);

  npy_intp dims[3] = {n_batch, n_channel, n_max_coord};
  PyObject* py_obj = PyArray_SimpleNewFromData(3, dims, NPY_FLOAT,
                                               extracted_activation);
  handle<> handle(py_obj);

  numeric::array arr(handle);

  free(activation);
  free(coords);

  return arr.copy();
}

object knn_fix_ref_conditional(object address_num,PyObject* query_points_, PyObject* musk_,int k,int cur_capacity)
{
  PyArrayObject* query_points = (PyArrayObject*) query_points_;
  PyArrayObject* musk = (PyArrayObject*) musk_;
  int n_query = query_points->dimensions[0];
  int dim     = query_points->dimensions[1];
  int address_num_c = extract<int>(address_num);

  float* query_points_c = new float[n_query * dim];
  bool* musk_c = new bool[cur_capacity];
  float* dist = new float[n_query * k];
  int* ind    = new int[n_query * k];

  for(int i = 0; i < n_query; i++) {
    for(int j = 0; j < dim; j++) {
      query_points_c[j + i*dim] =
          *(float*)PyArray_GETPTR2(query_points, i, j);
    }
  }

  for(int j = 0; j < cur_capacity; j++) {
      musk_c[j] =
          *(float*)PyArray_GETPTR1(query_points, j);
    }


  knn_cuda_fix_ref_conditional(address_num_c,dist, ind,query_points_c, musk_c,n_query, k,cur_capacity,dim);


  npy_intp dims[2] = {k, n_query};
  PyObject* py_obj_dist = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, dist);
  PyObject* py_obj_ind  = PyArray_SimpleNewFromData(2, dims, NPY_INT, ind);
  handle<> handle_dist(py_obj_dist);
  handle<> handle_ind(py_obj_ind);

  numeric::array arr_dist(handle_dist);
  numeric::array arr_ind(handle_ind);

  free(query_points_c);

  return make_tuple(arr_dist.copy(), arr_ind.copy());
}


// CUDA K-NN wrapper
// Takes features and retuns the distances and indices of the k-nearest
// neighboring features.
object knn_fix_ref(object address_num,PyObject* query_points_, int k,int cur_capacity)
{
  PyArrayObject* query_points = (PyArrayObject*) query_points_;
  int n_query = query_points->dimensions[0];
  int dim     = query_points->dimensions[1];
  int address_num_c = extract<int>(address_num);

  float* query_points_c = new float[n_query * dim];
  float* dist = new float[n_query * k];
  int* ind    = new int[n_query * k];


  for(int i = 0; i < n_query; i++) {
    for(int j = 0; j < dim; j++) {
      query_points_c[j + i*dim] =
          *(float*)PyArray_GETPTR2(query_points, i, j);
    }
  }


  knn_cuda_fix_ref(address_num_c,dist, ind,query_points_c, n_query, k,cur_capacity,dim);

  npy_intp dims[2] = {k, n_query};
  PyObject* py_obj_dist = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, dist);
  PyObject* py_obj_ind  = PyArray_SimpleNewFromData(2, dims, NPY_INT, ind);
  handle<> handle_dist(py_obj_dist);
  handle<> handle_ind(py_obj_ind);

  numeric::array arr_dist(handle_dist);
  numeric::array arr_ind(handle_ind);

  free(query_points_c);

  return make_tuple(arr_dist.copy(), arr_ind.copy());
}

int allocate(object size,object dims,object query_max,object k)
{
  int size_c = extract<int>(size);
  int k_c = extract<int>(k);
  int dims_c = extract<int>(dims);
  int query_max_c = extract<int>(query_max);
  allocate_cuda(address_count,size_c,dims_c,query_max_c,k_c);
  address_count+=1;
  printf("allocate finished.\n");
  return address_count-1;
}

void add(object address_num,object index,PyObject* ref_points_)
{
//printf("after here 0\n");
  int index_c = extract<int>(index);
  int address_num_c = extract<int>(address_num);
//  KNNAddress pointer_c = ArraytoAddress(pointer);
//printf("after here 1\n");
  PyArrayObject* ref_points = (PyArrayObject*) ref_points_;
//  printf("after here 2\n");
  int dim     = ref_points->dimensions[0];
//  printf("add dims:%d\n",dim);
  float* ref_points_c   = new float[dim];
//  printf("add dims:%d\n",dim);
  for(int j = 0; j < dim; j++) {
    ref_points_c[j] =
        *(float*)PyArray_GETPTR1(ref_points, j);
  }
//  printf("before add cuda\n");
  add_cuda(address_num_c,index_c,ref_points_c,dim);
//  printf("after add cuda\n");
  free(ref_points_c);
}



BOOST_PYTHON_MODULE(knn)
{
  import_array();
  numeric::array::set_module_and_type("numpy", "ndarray");
  def("knn", knn_fix_ref);
  def("knn_conditional", knn_fix_ref_conditional);

  def("allocate", allocate);
  def("add", add);
  def("extract", extract_feature);
}
