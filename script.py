from flask import abort, Flask, jsonify, request
from flask_healthz import healthz
from models.event_detection.EventDetector import EventDetector

app = Flask(__name__)

app.register_blueprint(healthz, url_prefix="/healthz")


def liveness():
    pass


def readiness():
    pass


app.config.update(
    HEALTHZ = {
        "live": app.name + ".liveness",
        "ready": app.name + ".readiness"
    }
)
event_detector = EventDetector()


@app.route('/', methods=['POST'])
def flask():
    if not request.json or not 'message' in request.json:
        print(request.json)
        abort(400)

    message = request.json['message']

    fig_cls, fig_cluster, description = event_detector.forward_batch(message)

    response = {'message': message, 'fig_cls': fig_cls, 'fig_cluster': fig_cluster,
                'descriptions': description}
    return jsonify(response), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5281)

