# 🎨 Vision-X ✨

**Advanced AI Image Analysis**
A powerful AI-powered tool that analyzes images to detect objects, scenes, colors, and text using the [Gemma3](https://ollama.com/library/gemma3) model.

## ✨ Features

- 🔍 Deep image analysis with AI
- 🎯 Object and scene detection
- 🌈 Color palette extraction
- 📝 Text recognition in images
- 🎭 Character and emotion detection
- 📊 Progress tracking and resumption
- 🚀 Batch processing support
- 💾 Automatic progress saving

## 🚀 Installation

1. Prerequisites:

   - [Python](https://www.python.org/downloads/) 3.8+
   - [Ollama](https://ollama.com/download) with [Gemma3](https://ollama.com/library/gemma3) model installed

2. Clone and install:

```bash
   git clone https://github.com/mahirox36/Vision-X.git
   cd image-analysis-tool IAT
   pip install -r requirements.txt
```

## 💻 Usage

### Command Line Options

```bash
# Analyze a single image
python main.py analyze -f "path/to/image.jpg"

# Process an entire directory
python main.py analyze -d "path/to/images" -b 10

# Show analysis details for a specific image
python main.py show "path/to/image.jpg"

# Show shortened analysis output
python main.py show -s "path/to/image.jpg"

# Save results in text format
python main.py analyze -d "path/to/images" --format txt
```

### Commands

#### analyze

Process images and generate analysis data.

| Argument           | Description                                | Default                           |
| ------------------ | ------------------------------------------ | --------------------------------- |
| `-f, --file`       | Single image to analyze                    | -                                 |
| `-d, --directory`  | Directory containing images                | -                                 |
| `-o, --output`     | Output file path                           | image_analysis_results.[json/txt] |
| `--format`         | Output format (json/txt)                   | json                              |
| `-b, --batch-size` | Number of images to process simultaneously | 10                                |
| `--no-progress`    | Disable progress bar                       | False                             |

#### show

Display analysis results for a previously analyzed image.

| Argument      | Description            |
| ------------- | ---------------------- |
| `-s, --short` | Show shortened output  |
| `-f, --full`  | Show full output       |
| `file`        | Path to the image file |

## 🔄 Progress Tracking

- Automatically saves progress after each image
- Resumes from last processed image if interrupted
- Tracks successful and failed analyses
- Saves detailed logs in `logs/image_analysis.log`

## 📝 Output Format

### JSON Output

```json
{
  "analysis_timestamp": "2024-...",
  "total_images": 100,
  "successful_analyses": 98,
  "failed_analyses": 2,
  "results": {
    "image1.jpg": {
      "summary": "...",
      "tags": ["..."],
      "objects": ["..."],
      "scene": "...",
      "colors": ["..."]
    }
  }
}
```

### Text Output

```text
=== image1.jpg ===
Summary: ...
Scene: ...
Objects: object1, object2, ...
```

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.
