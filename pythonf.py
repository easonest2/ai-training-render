from flask import Flask, render_template_string, send_from_directory, request, jsonify
import os

app = Flask(__name__)

# Directory to browse
BASE_DIR = os.path.expanduser(".")

# HTML Template with refresh button
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>File Explorer</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            margin: 0;
            padding: 0;
            background: var(--bg);
            color: var(--fg);
            transition: background 0.3s, color 0.3s;
        }
        :root {
            --bg: #1e1e1e;
            --fg: #f0f0f0;
            --accent: #0078d7;
            --card-bg: #2b2b2b;
        }
        .light {
            --bg: #f0f0f0;
            --fg: #1e1e1e;
            --accent: #0078d7;
            --card-bg: #ffffff;
        }
        header {
            padding: 10px;
            display: flex;
            justify-content: space-between;
            background: var(--card-bg);
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        h2 { margin: 0; }
        button {
            background: var(--accent);
            border: none;
            color: white;
            padding: 8px 14px;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s, background 0.3s;
        }
        button:hover {
            transform: scale(1.05);
            background: #005a9e;
        }
        .file-list {
            padding: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
        }
        .file-card {
            padding: 15px;
            border-radius: 12px;
            background: var(--card-bg);
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            transition: transform 0.2s, background 0.3s;
            cursor: pointer;
            text-align: center;
        }
        .file-card:hover {
            transform: translateY(-5px);
        }
        .file-card.selected {
            border: 2px solid var(--accent);
        }
        .download-btn {
            margin-top: 10px;
            display: none;
        }
        .file-card.selected .download-btn {
            display: block;
        }
    </style>
</head>
<body>
    <header>
        <h2>📂 File Explorer</h2>
        <div>
            <button onclick="toggleTheme()">🌙/☀️</button>
            <button onclick="refreshFiles()">🔄 Refresh</button>
        </div>
    </header>
    <div id="file-list" class="file-list">
        {% for file in files %}
        <div class="file-card" onclick="selectFile(this)">
            <div>{{ file }}</div>
            <a href="/download/{{ file }}" class="download-btn">
                <button>⬇ Download</button>
            </a>
        </div>
        {% endfor %}
    </div>

    <script>
        function selectFile(card) {
            document.querySelectorAll('.file-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
        }
        function toggleTheme() {
            document.body.classList.toggle("light");
        }
        async function refreshFiles() {
            const res = await fetch("/list");
            const files = await res.json();
            const container = document.getElementById("file-list");
            container.innerHTML = "";
            files.forEach(f => {
                const card = document.createElement("div");
                card.className = "file-card";
                card.onclick = () => selectFile(card);
                card.innerHTML = `<div>${f}</div><a href="/download/${f}" class="download-btn"><button>⬇ Download</button></a>`;
                container.appendChild(card);
            });
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    files = os.listdir(BASE_DIR)
    return render_template_string(HTML, files=files)

@app.route("/list")
def list_files():
    return jsonify(os.listdir(BASE_DIR))

@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(BASE_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
