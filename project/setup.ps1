Write-Output "Setting up ResuMentor Flask app..."

if (-Not (Test-Path "venv")) {
    Write-Output "Creating virtual environment..."
    python -m venv venv
}

Write-Output "Activating virtual environment..."
.\venv\Scripts\Activate.ps1

Write-Output "Upgrading pip..."
pip install --upgrade pip

if (Test-Path "requirements.txt") {
    Write-Output "Installing dependencies..."
    pip install -r requirements.txt
} else {
    Write-Output "No requirements.txt found. Skipping dependency installation."
}

Write-Output "Setup complete! You can now run: python main.py"
