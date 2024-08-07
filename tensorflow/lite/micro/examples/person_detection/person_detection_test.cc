/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"
#include "tensorflow/lite/micro/examples/person_detection/testdata/no_person_image_data.h"
#include "tensorflow/lite/micro/examples/person_detection/testdata/person_image_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/models/person_detect_model_data.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Got already on top
//#include "tensorflow/lite/micro/examples/person_detection/person_detect_model_data.h"
//#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
//#include "tensorflow/lite/micro/micro_interpreter.h"

// to add
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/system_setup.h"

// not sure if can go through
//#include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

constexpr int tensor_arena_size = 400 * 1024;
/*
// An area of memory to use for input, output, and intermediate arrays.
#if defined(XTENSA) && defined(VISION_P6)
constexpr int tensor_arena_size = 352 * 1024;
#else
constexpr int tensor_arena_size = 136 * 1024;
#endif  // defined(XTENSA) && defined(VISION_P6)
*/

//constexpr int tensor_arena_size = 93 * 1024;
alignas(16) static uint8_t tensor_arena[tensor_arena_size] = {0};
}  // namespace

/*
// Create an area of memory to use for input, output, and intermediate arrays.
#if defined(XTENSA) && defined(VISION_P6)
constexpr int tensor_arena_size = 352 * 1024;
#else
constexpr int tensor_arena_size = 136 * 1024;
#endif  // defined(XTENSA) && defined(VISION_P6)
uint8_t tensor_arena[tensor_arena_size];
*/

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInvoke) {

  // Follow PDM flow
  // Step 1: Get error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;


  // Step 2: Get model
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = ::tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  // Step 3: Get OpResolver to run the topology
  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  tflite::MicroMutableOpResolver<14> micro_op_resolver;

  // 10 operations needed
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddMinimum();
  micro_op_resolver.AddRelu();
  micro_op_resolver.AddMul();
  micro_op_resolver.AddRound();
  micro_op_resolver.AddSub();
  micro_op_resolver.AddAdd();

  // Keeping below to make compilation happy
  micro_op_resolver.AddAveragePool2D(tflite::Register_AVERAGE_POOL_2D_INT8());
  micro_op_resolver.AddConv2D(tflite::Register_CONV_2D_INT8());
  micro_op_resolver.AddDepthwiseConv2D(
      tflite::Register_DEPTHWISE_CONV_2D_INT8());
  micro_op_resolver.AddSoftmax(tflite::Register_SOFTMAX_INT8());
  //micro_op_resolver.AddReshape();

  // Step 4: Add interpreter
  // Build an interpreter to run the model with.
  tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena,
                                       tensor_arena_size);
  interpreter = &static_interpreter;
  //interpreter.AllocateTensors();
  // Add code to check allocate status
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return -1;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

  // Make sure the input has the properties we expect.
  TF_LITE_MICRO_EXPECT(input != nullptr);
  TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(kNumRows, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kNumCols, input->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(kNumChannels, input->dims->data[3]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, input->type);

  // Copy an image with a person into the memory area used for the input.
  TFLITE_DCHECK_EQ(input->bytes, static_cast<size_t>(g_person_image_data_size));
  memcpy(input->data.int8, g_person_image_data, input->bytes);

  // Step 5: Invoke
  // Run the model on this input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Step 6: Get output
  // Get the output from the model, and make sure it's the expected size and
  // type.
  TfLiteTensor* output = interpreter->output(0);
  TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(kCategoryCount, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, output->type);

  // Make sure that the expected "Person" score is higher than the other class.
  int8_t person_score = output->data.int8[kPersonIndex];
  int8_t no_person_score = output->data.int8[kNotAPersonIndex];
  MicroPrintf("person data.  person score: %d, no person score: %d\n",
              person_score, no_person_score);
  TF_LITE_MICRO_EXPECT_GT(person_score, no_person_score);

  memcpy(input->data.int8, g_no_person_image_data, input->bytes);

  // Run the model on this "No Person" input.
  invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Get the output from the model, and make sure it's the expected size and
  // type.
  output = interpreter->output(0);
  TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(kCategoryCount, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, output->type);

  // Make sure that the expected "No Person" score is higher.
  person_score = output->data.int8[kPersonIndex];
  no_person_score = output->data.int8[kNotAPersonIndex];
  MicroPrintf("no person data.  person score: %d, no person score: %d\n",
              person_score, no_person_score);
  TF_LITE_MICRO_EXPECT_GT(no_person_score, person_score);

  MicroPrintf("Ran successfully\n");
}

TF_LITE_MICRO_TESTS_END
