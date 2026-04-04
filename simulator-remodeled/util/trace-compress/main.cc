#include <iostream>
#include <string>

void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " --input <file.pb> --output <file.pb>"
            << " --from-version <4> --to-version <5>"
            << " [--func-id <N>]"
            << std::endl;
}

int main(int argc, char* argv[]) {
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
    return 1;
  }

  std::cout << "Converting " << input_path
            << " from v" << from_version << " to v" << to_version << std::endl;
  std::cerr << "Not yet implemented" << std::endl;
  return 1;
}
