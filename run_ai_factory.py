import traceback
from configparser import ConfigParser
import datetime
import multiprocessing
import pika
import socketio

from lib.common.common_util import logging
from process_procedure_all import predict_attention

config_file = "./config.ini"
config = ConfigParser()
config.read(config_file)

forward_secs = int(config['h5record_video'].getint('interval') / 1000) * config['h5record_video'].getint('chunk_num')
users_db = [f"7001_%03d" % i for i in range(1, 6)]
users_db.extend([f"7002_%03d" % i for i in range(1, 27)])


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class ProcessPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


if __name__ == "__main__":
    # connect to ws
    sio = socketio.Client()
    sio.connect(f"http://localhost:{config['h5record_video'].getint('tornado_port')}/record_video?uid=_pserver")


    @sio.event
    def connect():
        print("I'm connected!")


    @sio.event
    def connect_error():
        print("The connection failed!")


    @sio.event
    def disconnect():
        print("I'm disconnected!")


    # rabbitmq
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.exchange_declare(exchange='prediction', exchange_type='topic')

    result = channel.queue_declare('', exclusive=True, durable=True)
    queue_name = result.method.queue

    channel.queue_bind(
        exchange='prediction', queue=queue_name, routing_key='predict.start')

    print(' [*] Waiting for producer. To exit press CTRL+C')


    def callback(ch, method, properties, body):
        try:
            print(" [x] %r:%r" % (method.routing_key, body))
            user_id, start_time = str(body)[2:-1].split('*')
            end_time = (datetime.datetime.strptime(start_time, "%Y-%m-%d_%H:%M") + datetime.timedelta(
                seconds=forward_secs)).strftime('%Y-%m-%d_%H:%M')
            fixed_score, score, score_list = predict_attention(user_id, start_time, end_time)
            logging(
                f"{start_time}~{end_time}:\nmean - {fixed_score}\norigin_mean - {score}\nopenface_gru - {score_list[0]}\nopenface_lstm - {score_list[1]}\nopenpose_gru - {score_list[2]}\nopenpose_lstm - {score_list[3]}",
                f"logs/{user_id}/{start_time}_{end_time}")

            if score > 0:
                # TODO
                sio.emit('server_presult', ['user_id', f"{start_time} ~ {end_time}*{format(fixed_score, '.4f')}"],
                         namespace='/record_video')
                pass
                # redis_db1.set(user_id, f"{start_time} ~ {end_time}*{format(fixed_score, '.4f')}")

        except Exception:
            logging(
                f"[run_ai_factory.py][callback|id:{user_id}|stime:{start_time}][{datetime.datetime.now().strftime('%Y-%m-%d_%I:%M:%S')}]:"
                f"{traceback.format_exc()}",
                f"logs/error.log")


    channel.basic_consume(
        queue=queue_name, on_message_callback=callback, auto_ack=True)

    channel.start_consuming()
