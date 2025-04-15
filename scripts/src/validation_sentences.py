# Define target captions
TARGETS = [
    "A pedestrian signals the ego driver to stop, by putting their hand towards the ego driver.",
    "A person signals the driver to stop, by raising their hand towards the camera.",
]

# Define test cases and their expected similarity levels
VALID_LEVELS = {
"Extended": [
    "A pedestrian raises their hand towards the ego driver to stop traffic. They are looking scared and in need of help.",
    "A person puts their hand towards the ego driver to signal 'stop'. They are wearing a red t-shirt and blue pants.",
], "Equivalent": [
    "A pedestrian raises their hand towards the ego driver to stop traffic.",
    "A person puts their hand towards the ego driver to signal 'stop'.",
], "Partial": [
    "A person raises their hand towards the ego driver.",
    "A pedestrian signals the ego driver to stop.",
], "Slight": [
    "A human gestures to the ego driver.",
    "A person puts their hand out to the side.",
    "A pedestrian puts their hand up.",
], "Unrelated": [
    "The sky is blue and the sun is shining.",
    "A pedestrian is walking on the sidewalk.",
]}


IDEAL = {
    'Extended':     0.9,
    'Equivalent':   1.0,
    'Partial':      0.8,
    'Slight':       0.3,
    'Unrelated':    0.0
}