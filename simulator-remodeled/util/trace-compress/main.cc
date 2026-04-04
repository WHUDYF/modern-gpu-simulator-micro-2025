#include <fstream>
#include <iostream>
#include <string>
#include "trace_compress.h"
#include "threadblock.pb.h"

void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " --input <file.pb> --output <file.pb>"
            << " --from-version <4> --to-version <5>"
            << " [--func-id <N>]"
            << std::endl;
}

int main(int argc, char* argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::string input_path, output_path;
  int from_version = 4, to_version = 5;
  int func_id = 0;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) input_path = argv[++i];
    else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
    else if (arg == "--from-version" && i + 1 < argc) from_version = std::stoi(argv[++i]);
    else if (arg == "--to-version" && i + 1 < argc) to_version = std::stoi(argv[++i]);
    else if (arg == "--func-id" && i + 1 < argc) func_id = std::stoi(argv[++i]);
    else { print_usage(argv[0]); return 1; }
  }

  if (input_path.empty() || output_path.empty()) {
    print_usage(argv[0]);
    google::protobuf::ShutdownProtobufLibrary();
    return 1;
  }

  if (from_version != 4 || to_version != 5) {
    std::cerr << "Only v4->v5 conversion is supported" << std::endl;
    google::protobuf::ShutdownProtobufLibrary();
    return 1;
  }

  // Read v4 threadblock from input file
  dynamic_trace::threadblock tb;
  {
    std::ifstream in(input_path, std::ios::binary);
    if (!in.is_open()) {
      std::cerr << "Cannot open input file: " << input_path << std::endl;
      google::protobuf::ShutdownProtobufLibrary();
      return 1;
    }
    if (!tb.ParseFromIstream(&in)) {
      std::cerr << "Failed to parse input as threadblock: " << input_path << std::endl;
      google::protobuf::ShutdownProtobufLibrary();
      return 1;
    }
  }

  // Encode v4 -> v5
  dynamic_trace::compressed_threadblock ctb;
  if (!encode_v4_to_v5(tb, &ctb, func_id)) {
    std::cerr << "Encoding failed" << std::endl;
    google::protobuf::ShutdownProtobufLibrary();
    return 1;
  }

  // Write compressed output
  {
    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open()) {
      std::cerr << "Cannot open output file: " << output_path << std::endl;
      google::protobuf::ShutdownProtobufLibrary();
      return 1;
    }
    if (!ctb.SerializeToOstream(&out)) {
      std::cerr << "Failed to serialize compressed threadblock" << std::endl;
      google::protobuf::ShutdownProtobufLibrary();
      return 1;
    }
  }

  size_t original_size = tb.ByteSizeLong();
  size_t compressed_size = ctb.ByteSizeLong();
  double ratio = (original_size > 0)
      ? static_cast<double>(compressed_size) / static_cast<double>(original_size)
      : 0.0;

  std::cout << "Original size:   " << original_size << " bytes" << std::endl;
  std::cout << "Compressed size: " << compressed_size << " bytes" << std::endl;
  std::cout << "Ratio:           " << ratio << std::endl;

  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
