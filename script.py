from event_extractor import EventExtractor
from flask import abort, Flask, jsonify, request
from flask_healthz import healthz
from models.entity_linking.BLINKEntityLinker import BLINKEntityLinker
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
event_argument_extractor = BLINKEntityLinker()
event_extractor = EventExtractor(event_detector=event_detector, event_argument_extractor=event_argument_extractor)


@app.route('/', methods=['POST'])
def flask():
    authenticated = False

    if 'key' in request.json:
        key = request.json['key']
        if (key == '32T82GWPSGDJTKFN'):
            authenticated = True

    if (authenticated == False):
        abort(401)

    if not request.json or not 'message' in request.json:
        abort(400)

    message = request.json['message']

    output = event_extractor.infer(message)

    response = {'message': message, 'event type': output[0], 'event arguments': output[1],
                'event graph': output[2]}
    return jsonify(response), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5278)

