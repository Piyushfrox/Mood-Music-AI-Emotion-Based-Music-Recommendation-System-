import webbrowser

def play_music(emotion):
    if emotion == "happy":
        url = "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC"   # Happy playlist
    elif emotion == "sad":
        url = "https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1"   # Sad playlist
    else:
        url = "https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0"   # Neutral/Chill playlist

    print(f"🎵 Playing music for: {emotion}")
    webbrowser.open(url)
