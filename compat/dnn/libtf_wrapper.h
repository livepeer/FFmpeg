/*
 * Copyright (c) 2021 Akul Penugonda
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Wrapper header containing function pointer types for functions in libtensorflow
 */

#include <tensorflow/c/c_api.h>

#ifdef _WIN32
#include <windows.h>
#define TF_LOAD_FUNC(path) LoadLibrary(path)
#define TF_SYM_FUNC(lib, sym) GetProcAddress(lib, sym)
#define TF_ERROR_FUNC() GetLastError()
#define TF_FREE_FUNC(lib) FreeLibrary(lib)
#define TF_LIBNAME "tensorflow.dll"
#else
#include <dlfcn.h>
#define TF_LOAD_FUNC(path) dlopen(path, RTLD_LAZY | RTLD_GLOBAL)
#define TF_SYM_FUNC(lib, sym) dlsym(lib, sym)
#define TF_ERROR_FUNC() dlerror()
#define TF_FREE_FUNC(lib) dlclose(lib)
#define TF_LIBNAME "libtensorflow.so"
#endif

typedef const char* ( *PFN_TF_Version)(void);

typedef TF_Buffer* ( *PFN_TF_NewBufferFromString)(const void* proto,
                                                        size_t proto_len);
typedef TF_Buffer* ( *PFN_TF_NewBuffer)(void);

typedef void ( *PFN_TF_DeleteBuffer)(TF_Buffer*);

typedef TF_Buffer ( *PFN_TF_GetBuffer)(TF_Buffer* buffer);

typedef TF_SessionOptions* ( *PFN_TF_NewSessionOptions)(void);

typedef void ( *PFN_TF_SetTarget)(TF_SessionOptions* options,
                                        const char* target);

typedef void ( *PFN_TF_SetConfig)(TF_SessionOptions* options,
                                        const void* proto, size_t proto_len,
                                        TF_Status* status);

typedef void ( *PFN_TF_DeleteSessionOptions)(TF_SessionOptions*);

typedef TF_Graph* ( *PFN_TF_NewGraph)(void);

typedef void ( *PFN_TF_DeleteGraph)(TF_Graph*);


typedef void ( *PFN_TF_GraphSetTensorShape)(TF_Graph* graph,
                                                  TF_Output output,
                                                  const int64_t* dims,
                                                  const int num_dims,
                                                  TF_Status* status);

typedef int ( *PFN_TF_GraphGetTensorNumDims)(TF_Graph* graph,
                                                   TF_Output output,
                                                   TF_Status* status);

typedef void ( *PFN_TF_GraphGetTensorShape)(TF_Graph* graph,
                                                  TF_Output output,
                                                  int64_t* dims, int num_dims,
                                                  TF_Status* status);

typedef TF_OperationDescription* ( *PFN_TF_NewOperation)(
    TF_Graph* graph, const char* op_type, const char* oper_name);

typedef void ( *PFN_TF_SetDevice)(TF_OperationDescription* desc,
                                        const char* device);


typedef void ( *PFN_TF_AddInput)(TF_OperationDescription* desc,
                                       TF_Output input);

typedef void ( *PFN_TF_AddInputList)(TF_OperationDescription* desc,
                                           const TF_Output* inputs,
                                           int num_inputs);

typedef void ( *PFN_TF_AddControlInput)(TF_OperationDescription* desc,
                                              TF_Operation* input);

typedef void ( *PFN_TF_ColocateWith)(TF_OperationDescription* desc,
                                           TF_Operation* op);


typedef void ( *PFN_TF_SetAttrString)(TF_OperationDescription* desc,
                                            const char* attr_name,
                                            const void* value, size_t length);
typedef void ( *PFN_TF_SetAttrStringList)(TF_OperationDescription* desc,
                                                const char* attr_name,
                                                const void* const* values,
                                                const size_t* lengths,
                                                int num_values);
typedef void ( *PFN_TF_SetAttrInt)(TF_OperationDescription* desc,
                                         const char* attr_name, int64_t value);
typedef void ( *PFN_TF_SetAttrIntList)(TF_OperationDescription* desc,
                                             const char* attr_name,
                                             const int64_t* values,
                                             int num_values);
typedef void ( *PFN_TF_SetAttrFloat)(TF_OperationDescription* desc,
                                           const char* attr_name, float value);
typedef void ( *PFN_TF_SetAttrFloatList)(TF_OperationDescription* desc,
                                               const char* attr_name,
                                               const float* values,
                                               int num_values);
typedef void ( *PFN_TF_SetAttrBool)(TF_OperationDescription* desc,
                                          const char* attr_name,
                                          unsigned char value);
typedef void ( *PFN_TF_SetAttrBoolList)(TF_OperationDescription* desc,
                                              const char* attr_name,
                                              const unsigned char* values,
                                              int num_values);
typedef void ( *PFN_TF_SetAttrType)(TF_OperationDescription* desc,
                                          const char* attr_name,
                                          TF_DataType value);
typedef void ( *PFN_TF_SetAttrTypeList)(TF_OperationDescription* desc,
                                              const char* attr_name,
                                              const TF_DataType* values,
                                              int num_values);
typedef void ( *PFN_TF_SetAttrPlaceholder)(TF_OperationDescription* desc,
                                                 const char* attr_name,
                                                 const char* placeholder);

typedef void ( *PFN_TF_SetAttrFuncName)(TF_OperationDescription* desc,
                                              const char* attr_name,
                                              const char* value, size_t length);

typedef void ( *PFN_TF_SetAttrShape)(TF_OperationDescription* desc,
                                           const char* attr_name,
                                           const int64_t* dims, int num_dims);
typedef void ( *PFN_TF_SetAttrShapeList)(TF_OperationDescription* desc,
                                               const char* attr_name,
                                               const int64_t* const* dims,
                                               const int* num_dims,
                                               int num_shapes);
typedef void ( *PFN_TF_SetAttrTensorShapeProto)(
    TF_OperationDescription* desc, const char* attr_name, const void* proto,
    size_t proto_len, TF_Status* status);
typedef void ( *PFN_TF_SetAttrTensorShapeProtoList)(
    TF_OperationDescription* desc, const char* attr_name,
    const void* const* protos, const size_t* proto_lens, int num_shapes,
    TF_Status* status);

typedef void ( *PFN_TF_SetAttrTensor)(TF_OperationDescription* desc,
                                            const char* attr_name,
                                            TF_Tensor* value,
                                            TF_Status* status);
typedef void ( *PFN_TF_SetAttrTensorList)(TF_OperationDescription* desc,
                                                const char* attr_name,
                                                TF_Tensor* const* values,
                                                int num_values,
                                                TF_Status* status);

typedef void ( *PFN_TF_SetAttrValueProto)(TF_OperationDescription* desc,
                                                const char* attr_name,
                                                const void* proto,
                                                size_t proto_len,
                                                TF_Status* status);

typedef TF_Operation* ( *PFN_TF_FinishOperation)(
    TF_OperationDescription* desc, TF_Status* status);


typedef const char* ( *PFN_TF_OperationName)(TF_Operation* oper);
typedef const char* ( *PFN_TF_OperationOpType)(TF_Operation* oper);
typedef const char* ( *PFN_TF_OperationDevice)(TF_Operation* oper);

typedef int ( *PFN_TF_OperationNumOutputs)(TF_Operation* oper);
typedef TF_DataType ( *PFN_TF_OperationOutputType)(TF_Output oper_out);
typedef int ( *PFN_TF_OperationOutputListLength)(TF_Operation* oper,
                                                       const char* arg_name,
                                                       TF_Status* status);

typedef int ( *PFN_TF_OperationNumInputs)(TF_Operation* oper);
typedef TF_DataType ( *PFN_TF_OperationInputType)(TF_Input oper_in);
typedef int ( *PFN_TF_OperationInputListLength)(TF_Operation* oper,
                                                      const char* arg_name,
                                                      TF_Status* status);

typedef TF_Output ( *PFN_TF_OperationInput)(TF_Input oper_in);

typedef void ( *PFN_TF_OperationAllInputs)(TF_Operation* oper,
                                                 TF_Output* inputs,
                                                 int max_inputs);

typedef int ( *PFN_TF_OperationOutputNumConsumers)(TF_Output oper_out);

typedef int ( *PFN_TF_OperationOutputConsumers)(TF_Output oper_out,
                                                      TF_Input* consumers,
                                                      int max_consumers);

typedef int ( *PFN_TF_OperationNumControlInputs)(TF_Operation* oper);

typedef int ( *PFN_TF_OperationGetControlInputs)(
    TF_Operation* oper, TF_Operation** control_inputs, int max_control_inputs);

typedef int ( *PFN_TF_OperationNumControlOutputs)(TF_Operation* oper);

typedef int ( *PFN_TF_OperationGetControlOutputs)(
    TF_Operation* oper, TF_Operation** control_outputs,
    int max_control_outputs);



typedef TF_AttrMetadata ( *PFN_TF_OperationGetAttrMetadata)(
    TF_Operation* oper, const char* attr_name, TF_Status* status);

typedef void ( *PFN_TF_OperationGetAttrString)(TF_Operation* oper,
                                                     const char* attr_name,
                                                     void* value,
                                                     size_t max_length,
                                                     TF_Status* status);

typedef void ( *PFN_TF_OperationGetAttrStringList)(
    TF_Operation* oper, const char* attr_name, void** values, size_t* lengths,
    int max_values, void* storage, size_t storage_size, TF_Status* status);

typedef void ( *PFN_TF_OperationGetAttrInt)(TF_Operation* oper,
                                                  const char* attr_name,
                                                  int64_t* value,
                                                  TF_Status* status);

typedef void ( *PFN_TF_OperationGetAttrIntList)(TF_Operation* oper,
                                                      const char* attr_name,
                                                      int64_t* values,
                                                      int max_values,
                                                      TF_Status* status);

typedef void ( *PFN_TF_OperationGetAttrFloat)(TF_Operation* oper,
                                                    const char* attr_name,
                                                    float* value,
                                                    TF_Status* status);

typedef void ( *PFN_TF_OperationGetAttrFloatList)(TF_Operation* oper,
                                                        const char* attr_name,
                                                        float* values,
                                                        int max_values,
                                                        TF_Status* status);

typedef void ( *PFN_TF_OperationGetAttrBool)(TF_Operation* oper,
                                                   const char* attr_name,
                                                   unsigned char* value,
                                                   TF_Status* status);

typedef void ( *PFN_TF_OperationGetAttrBoolList)(TF_Operation* oper,
                                                       const char* attr_name,
                                                       unsigned char* values,
                                                       int max_values,
                                                       TF_Status* status);

typedef void ( *PFN_TF_OperationGetAttrType)(TF_Operation* oper,
                                                   const char* attr_name,
                                                   TF_DataType* value,
                                                   TF_Status* status);

typedef void ( *PFN_TF_OperationGetAttrTypeList)(TF_Operation* oper,
                                                       const char* attr_name,
                                                       TF_DataType* values,
                                                       int max_values,
                                                       TF_Status* status);

typedef void ( *PFN_TF_OperationGetAttrShape)(TF_Operation* oper,
                                                    const char* attr_name,
                                                    int64_t* value,
                                                    int num_dims,
                                                    TF_Status* status);

typedef void ( *PFN_TF_OperationGetAttrShapeList)(
    TF_Operation* oper, const char* attr_name, int64_t** dims, int* num_dims,
    int num_shapes, int64_t* storage, int storage_size, TF_Status* status);

typedef void ( *PFN_TF_OperationGetAttrTensorShapeProto)(
    TF_Operation* oper, const char* attr_name, TF_Buffer* value,
    TF_Status* status);

typedef void ( *PFN_TF_OperationGetAttrTensorShapeProtoList)(
    TF_Operation* oper, const char* attr_name, TF_Buffer** values,
    int max_values, TF_Status* status);

typedef void ( *PFN_TF_OperationGetAttrTensor)(TF_Operation* oper,
                                                     const char* attr_name,
                                                     TF_Tensor** value,
                                                     TF_Status* status);

typedef void ( *PFN_TF_OperationGetAttrTensorList)(TF_Operation* oper,
                                                         const char* attr_name,
                                                         TF_Tensor** values,
                                                         int max_values,
                                                         TF_Status* status);

typedef void ( *PFN_TF_OperationGetAttrValueProto)(
    TF_Operation* oper, const char* attr_name, TF_Buffer* output_attr_value,
    TF_Status* status);

typedef TF_Operation* ( *PFN_TF_GraphOperationByName)(
    TF_Graph* graph, const char* oper_name);

typedef TF_Operation* ( *PFN_TF_GraphNextOperation)(TF_Graph* graph,
                                                          size_t* pos);

typedef void ( *PFN_TF_GraphToGraphDef)(TF_Graph* graph,
                                              TF_Buffer* output_graph_def,
                                              TF_Status* status);

typedef void ( *PFN_TF_GraphGetOpDef)(TF_Graph* graph,
                                            const char* op_name,
                                            TF_Buffer* output_op_def,
                                            TF_Status* status);

typedef void ( *PFN_TF_GraphVersions)(TF_Graph* graph,
                                            TF_Buffer* output_version_def,
                                            TF_Status* status);

typedef TF_ImportGraphDefOptions* ( *PFN_TF_NewImportGraphDefOptions)(
    void);
typedef void ( *PFN_TF_DeleteImportGraphDefOptions)(
    TF_ImportGraphDefOptions* opts);

typedef void ( *PFN_TF_ImportGraphDefOptionsSetPrefix)(
    TF_ImportGraphDefOptions* opts, const char* prefix);

typedef void ( *PFN_TF_ImportGraphDefOptionsSetDefaultDevice)(
    TF_ImportGraphDefOptions* opts, const char* device);

typedef void ( *PFN_TF_ImportGraphDefOptionsSetUniquifyNames)(
    TF_ImportGraphDefOptions* opts, unsigned char uniquify_names);

typedef void ( *PFN_TF_ImportGraphDefOptionsSetUniquifyPrefix)(
    TF_ImportGraphDefOptions* opts, unsigned char uniquify_prefix);

typedef void ( *PFN_TF_ImportGraphDefOptionsAddInputMapping)(
    TF_ImportGraphDefOptions* opts, const char* src_name, int src_index,
    TF_Output dst);

typedef void ( *PFN_TF_ImportGraphDefOptionsRemapControlDependency)(
    TF_ImportGraphDefOptions* opts, const char* src_name, TF_Operation* dst);

typedef void ( *PFN_TF_ImportGraphDefOptionsAddControlDependency)(
    TF_ImportGraphDefOptions* opts, TF_Operation* oper);

typedef void ( *PFN_TF_ImportGraphDefOptionsAddReturnOutput)(
    TF_ImportGraphDefOptions* opts, const char* oper_name, int index);

typedef int ( *PFN_TF_ImportGraphDefOptionsNumReturnOutputs)(
    const TF_ImportGraphDefOptions* opts);

typedef void ( *PFN_TF_ImportGraphDefOptionsAddReturnOperation)(
    TF_ImportGraphDefOptions* opts, const char* oper_name);

typedef int ( *PFN_TF_ImportGraphDefOptionsNumReturnOperations)(
    const TF_ImportGraphDefOptions* opts);


typedef void ( *PFN_TF_ImportGraphDefResultsReturnOutputs)(
    TF_ImportGraphDefResults* results, int* num_outputs, TF_Output** outputs);

typedef void ( *PFN_TF_ImportGraphDefResultsReturnOperations)(
    TF_ImportGraphDefResults* results, int* num_opers, TF_Operation*** opers);

typedef void ( *PFN_TF_ImportGraphDefResultsMissingUnusedInputMappings)(
    TF_ImportGraphDefResults* results, int* num_missing_unused_input_mappings,
    const char*** src_names, int** src_indexes);

typedef void ( *PFN_TF_DeleteImportGraphDefResults)(
    TF_ImportGraphDefResults* results);

typedef TF_ImportGraphDefResults**
( *PFN_TF_GraphImportGraphDefWithResults)(TF_Graph* graph, const TF_Buffer* graph_def,
                                  const TF_ImportGraphDefOptions* options,
                                  TF_Status* status);

typedef void ( *PFN_TF_GraphImportGraphDefWithReturnOutputs)(
    TF_Graph* graph, const TF_Buffer* graph_def,
    const TF_ImportGraphDefOptions* options, TF_Output* return_outputs,
    int num_return_outputs, TF_Status* status);

typedef void ( *PFN_TF_GraphImportGraphDef)(
    TF_Graph* graph, const TF_Buffer* graph_def,
    const TF_ImportGraphDefOptions* options, TF_Status* status);

typedef void ( *PFN_TF_GraphCopyFunction)(TF_Graph* g,
                                                const TF_Function* func,
                                                const TF_Function* grad,
                                                TF_Status* status);

typedef int ( *PFN_TF_GraphNumFunctions)(TF_Graph* g);

typedef int ( *PFN_TF_GraphGetFunctions)(TF_Graph* g, TF_Function** funcs,
                                               int max_func, TF_Status* status);


typedef void ( *PFN_TF_OperationToNodeDef)(TF_Operation* oper,
                                                 TF_Buffer* output_node_def,
                                                 TF_Status* status);



typedef TF_WhileParams ( *PFN_TF_NewWhile)(TF_Graph* g, TF_Output* inputs,
                                                 int ninputs,
                                                 TF_Status* status);

typedef void ( *PFN_TF_FinishWhile)(const TF_WhileParams* params,
                                          TF_Status* status,
                                          TF_Output* outputs);

typedef void ( *PFN_TF_AbortWhile)(const TF_WhileParams* params);




typedef TF_Function* ( *PFN_TF_GraphToFunction)(
    const TF_Graph* fn_body, const char* fn_name,
    unsigned char append_hash_to_fn_name, int num_opers,
    const TF_Operation* const* opers, int ninputs, const TF_Output* inputs,
    int noutputs, const TF_Output* outputs, const char* const* output_names,
    const TF_FunctionOptions* opts, const char* description, TF_Status* status);

typedef TF_Function* ( *PFN_TF_GraphToFunctionWithControlOutputs)(
    const TF_Graph* fn_body, const char* fn_name,
    unsigned char append_hash_to_fn_name, int num_opers,
    const TF_Operation* const* opers, int ninputs, const TF_Output* inputs,
    int noutputs, const TF_Output* outputs, const char* const* output_names,
    int ncontrol_outputs, const TF_Operation* const* control_outputs,
    const char* const* control_output_names, const TF_FunctionOptions* opts,
    const char* description, TF_Status* status);

typedef const char* ( *PFN_TF_FunctionName)(TF_Function* func);

typedef void ( *PFN_TF_FunctionToFunctionDef)(TF_Function* func,
                                                    TF_Buffer* output_func_def,
                                                    TF_Status* status);

typedef TF_Function* ( *PFN_TF_FunctionImportFunctionDef)(
    const void* proto, size_t proto_len, TF_Status* status);

typedef void ( *PFN_TF_FunctionSetAttrValueProto)(TF_Function* func,
                                                        const char* attr_name,
                                                        const void* proto,
                                                        size_t proto_len,
                                                        TF_Status* status);

typedef void ( *PFN_TF_FunctionGetAttrValueProto)(
    TF_Function* func, const char* attr_name, TF_Buffer* output_attr_value,
    TF_Status* status);

typedef void ( *PFN_TF_DeleteFunction)(TF_Function* func);

typedef unsigned char ( *PFN_TF_TryEvaluateConstant)(TF_Graph* graph,
                                                           TF_Output output,
                                                           TF_Tensor** result,
                                                           TF_Status* status);




typedef TF_Session* ( *PFN_TF_NewSession)(TF_Graph* graph,
                                                const TF_SessionOptions* opts,
                                                TF_Status* status);

typedef TF_Session* ( *PFN_TF_LoadSessionFromSavedModel)(
    const TF_SessionOptions* session_options, const TF_Buffer* run_options,
    const char* export_dir, const char* const* tags, int tags_len,
    TF_Graph* graph, TF_Buffer* meta_graph_def, TF_Status* status);

typedef void ( *PFN_TF_CloseSession)(TF_Session*, TF_Status* status);

typedef void ( *PFN_TF_DeleteSession)(TF_Session*, TF_Status* status);

typedef void ( *PFN_TF_SessionRun)(
    TF_Session* session,
    // RunOptions
    const TF_Buffer* run_options,
    // Input tensors
    const TF_Output* inputs, TF_Tensor* const* input_values, int ninputs,
    // Output tensors
    const TF_Output* outputs, TF_Tensor** output_values, int noutputs,
    // Target operations
    const TF_Operation* const* target_opers, int ntargets,
    // RunMetadata
    TF_Buffer* run_metadata,
    // Output status
    TF_Status*);

typedef void ( *PFN_TF_SessionPRunSetup)(
    TF_Session*,
    // Input names
    const TF_Output* inputs, int ninputs,
    // Output names
    const TF_Output* outputs, int noutputs,
    // Target operations
    const TF_Operation* const* target_opers, int ntargets,
    // Output handle
    const char** handle,
    // Output status
    TF_Status*);

typedef void ( *PFN_TF_SessionPRun)(
    TF_Session*, const char* handle,
    // Input tensors
    const TF_Output* inputs, TF_Tensor* const* input_values, int ninputs,
    // Output tensors
    const TF_Output* outputs, TF_Tensor** output_values, int noutputs,
    // Target operations
    const TF_Operation* const* target_opers, int ntargets,
    // Output status
    TF_Status*);

typedef void ( *PFN_TF_DeletePRunHandle)(const char* handle);



typedef TF_DeprecatedSession** ( *PFN_TF_NewDeprecatedSession)(
    const TF_SessionOptions*, TF_Status* status);
typedef void ( *PFN_TF_CloseDeprecatedSession)(TF_DeprecatedSession*,
                                                     TF_Status* status);
typedef void ( *PFN_TF_DeleteDeprecatedSession)(TF_DeprecatedSession*,
                                                      TF_Status* status);
typedef void ( *PFN_TF_Reset)(const TF_SessionOptions* opt,
                                    const char** containers, int ncontainers,
                                    TF_Status* status);
typedef void ( *PFN_TF_ExtendGraph)(TF_DeprecatedSession*,
                                          const void* proto, size_t proto_len,
                                          TF_Status*);

typedef void ( *PFN_TF_Run)(TF_DeprecatedSession*,
                                  const TF_Buffer* run_options,
                                  const char** input_names, TF_Tensor** inputs,
                                  int ninputs, const char** output_names,
                                  TF_Tensor** outputs, int noutputs,
                                  const char** target_oper_names, int ntargets,
                                  TF_Buffer* run_metadata, TF_Status*);

typedef void ( *PFN_TF_PRunSetup)(TF_DeprecatedSession*,
                                        const char** input_names, int ninputs,
                                        const char** output_names, int noutputs,
                                        const char** target_oper_names,
                                        int ntargets, const char** handle,
                                        TF_Status*);

typedef void ( *PFN_TF_PRun)(TF_DeprecatedSession*, const char* handle,
                                   const char** input_names, TF_Tensor** inputs,
                                   int ninputs, const char** output_names,
                                   TF_Tensor** outputs, int noutputs,
                                   const char** target_oper_names, int ntargets,
                                   TF_Status*);


typedef TF_DeviceList* ( *PFN_TF_SessionListDevices)(TF_Session* session,
                                                           TF_Status* status);

typedef TF_DeviceList* ( *PFN_TF_DeprecatedSessionListDevices)(
    TF_DeprecatedSession* session, TF_Status* status);

typedef void ( *PFN_TF_DeleteDeviceList)(TF_DeviceList* list);

typedef int ( *PFN_TF_DeviceListCount)(const TF_DeviceList* list);

typedef const char* ( *PFN_TF_DeviceListName)(const TF_DeviceList* list,
                                                    int index,
                                                    TF_Status* status);

typedef const char* ( *PFN_TF_DeviceListType)(const TF_DeviceList* list,
                                                    int index,
                                                    TF_Status* status);

typedef int64_t ( *PFN_TF_DeviceListMemoryBytes)(
    const TF_DeviceList* list, int index, TF_Status* status);

typedef uint64_t ( *PFN_TF_DeviceListIncarnation)(
    const TF_DeviceList* list, int index, TF_Status* status);




typedef TF_Library* ( *PFN_TF_LoadLibrary)(const char* library_filename,
                                                 TF_Status* status);

typedef TF_Buffer ( *PFN_TF_GetOpList)(TF_Library* lib_handle);

typedef void ( *PFN_TF_DeleteLibraryHandle)(TF_Library* lib_handle);

typedef TF_Buffer* ( *PFN_TF_GetAllOpList)(void);


typedef TF_ApiDefMap* ( *PFN_TF_NewApiDefMap)(TF_Buffer* op_list_buffer,
                                                    TF_Status* status);

typedef void ( *PFN_TF_DeleteApiDefMap)(TF_ApiDefMap* apimap);

typedef void ( *PFN_TF_ApiDefMapPut)(TF_ApiDefMap* api_def_map,
                                           const char* text, size_t text_len,
                                           TF_Status* status);

typedef TF_Buffer* ( *PFN_TF_ApiDefMapGet)(TF_ApiDefMap* api_def_map,
                                                 const char* name,
                                                 size_t name_len,
                                                 TF_Status* status);


typedef TF_Buffer* ( *PFN_TF_GetAllRegisteredKernels)(TF_Status* status);

typedef TF_Buffer* ( *PFN_TF_GetRegisteredKernelsForOp)(
    const char* name, TF_Status* status);

typedef void ( *PFN_TF_UpdateEdge)(TF_Graph* graph, TF_Output new_src,
                                         TF_Input dst, TF_Status* status);




typedef TF_Server* ( *PFN_TF_NewServer)(const void* proto,
                                              size_t proto_len,
                                              TF_Status* status);

typedef void ( *PFN_TF_ServerStart)(TF_Server* server, TF_Status* status);

typedef void ( *PFN_TF_ServerStop)(TF_Server* server, TF_Status* status);

typedef void ( *PFN_TF_ServerJoin)(TF_Server* server, TF_Status* status);

typedef const char* ( *PFN_TF_ServerTarget)(TF_Server* server);

typedef void ( *PFN_TF_DeleteServer)(TF_Server* server);

typedef void ( *PFN_TF_RegisterLogListener)(
    void (*listener)(const char*));

typedef void ( *PFN_TF_RegisterFilesystemPlugin)(
    const char* plugin_filename, TF_Status* status);


typedef TF_Tensor* ( *PFN_TF_NewTensor)(
    TF_DataType, const int64_t* dims, int num_dims, void* data, size_t len,
    void (*deallocator)(void* data, size_t len, void* arg),
    void* deallocator_arg);

typedef TF_Tensor* ( *PFN_TF_AllocateTensor)(TF_DataType,
                                                   const int64_t* dims,
                                                   int num_dims, size_t len);

typedef TF_Tensor* ( *PFN_TF_TensorMaybeMove)(TF_Tensor* tensor);

typedef void ( *PFN_TF_DeleteTensor)(TF_Tensor*);

typedef TF_DataType ( *PFN_TF_TensorType)(const TF_Tensor*);

typedef int ( *PFN_TF_NumDims)(const TF_Tensor*);

typedef int64_t ( *PFN_TF_Dim)(const TF_Tensor* tensor, int dim_index);

typedef size_t ( *PFN_TF_TensorByteSize)(const TF_Tensor*);

typedef void* ( *PFN_TF_TensorData)(const TF_Tensor*);

typedef int64_t ( *PFN_TF_TensorElementCount)(const TF_Tensor* tensor);

typedef void ( *PFN_TF_TensorBitcastFrom)(const TF_Tensor* from,
                                                TF_DataType type, TF_Tensor* to,
                                                const int64_t* new_dims,
                                                int num_new_dims,
                                                TF_Status* status);

typedef bool ( *PFN_TF_TensorIsAligned)(const TF_Tensor*);

typedef TF_Status* ( *PFN_TF_NewStatus)(void);

typedef void ( *PFN_TF_DeleteStatus)(TF_Status*);

typedef void ( *PFN_TF_SetStatus)(TF_Status* s, TF_Code code,
                                        const char* msg);

typedef void ( *PFN_TF_SetStatusFromIOError)(TF_Status* s, int error_code,
                                                   const char* context);

typedef TF_Code ( *PFN_TF_GetCode)(const TF_Status* s);

typedef const char* ( *PFN_TF_Message)(const TF_Status* s);

typedef size_t ( *PFN_TF_DataTypeSize)(TF_DataType dt);

#define FN_LIST(MACRO)					    \
    MACRO(TF_SessionRun)				    \
    MACRO(TF_GetCode)					    \
    MACRO(TF_Dim)					    \
    MACRO(TF_TensorData)				    \
    MACRO(TF_TensorType)				    \
    MACRO(TF_CloseSession)				    \
    MACRO(TF_NewSession)				    \
    MACRO(TF_DeleteSession)				    \
    MACRO(TF_DeleteStatus)				    \
    MACRO(TF_DeleteGraph)				    \
    MACRO(TF_NewGraph)					    \
    MACRO(TF_DeleteTensor)				    \
    MACRO(TF_GraphOperationByName)			    \
    MACRO(TF_OperationOutputType)			    \
    MACRO(TF_NewStatus)				    \
    MACRO(TF_GraphGetTensorShape)			    \
    MACRO(TF_NewOperation)				    \
    MACRO(TF_SetAttrType)				    \
    MACRO(TF_SetAttrShape)				    \
    MACRO(TF_FinishOperation)				    \
    MACRO(TF_AllocateTensor)				    \
    MACRO(TF_SetAttrTensor)				    \
    MACRO(TF_AddInput)					    \
    MACRO(TF_SetAttrInt)				    \
    MACRO(TF_DataTypeSize)				    \
    MACRO(TF_SetAttrString)				    \
    MACRO(TF_SetAttrIntList)				    \
    MACRO(TF_NewSessionOptions)			    \
    MACRO(TF_SetConfig)				    \
    MACRO(TF_DeleteSessionOptions)			    \
    MACRO(TF_NewImportGraphDefOptions)			    \
    MACRO(TF_ImportGraphDefOptionsSetDefaultDevice)	    \
    MACRO(TF_GraphImportGraphDef)			    \
    MACRO(TF_DeleteImportGraphDefOptions)		    \
    MACRO(TF_DeleteBuffer)				    \
    MACRO(TF_NewBuffer)				    \
    MACRO(TF_Version)

#define TF_LOAD_SYMBOL(sym) tf_model->tffns->sym = (PFN_##sym)TF_SYM_FUNC(tf_model->libtensorflow, #sym);

#define PFN_DEF(name)  \
    PFN_##name name;

typedef struct TFFunctions {
    FN_LIST(PFN_DEF)
} TFFunctions;
