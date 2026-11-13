import json


def completed_summary(text="Завтра встреча в семь."):
    return {
        "status": "completed",
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps({"summary": text}),
                        "annotations": [],
                    }
                ],
            }
        ],
    }
