# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import sgd_pb2 as sgd__pb2


class SGDStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.ComputeTask = channel.unary_unary(
        '/stochastic.SGD/ComputeTask',
        request_serializer=sgd__pb2.LWB.SerializeToString,
        response_deserializer=sgd__pb2.Update.FromString,
        )


class SGDServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def ComputeTask(self, request, context):
    """A joining client tells the server that he is ready to compute
    the Server respond with a computation task if it is okay.
    rpc Ready(ReadyRequest) returns (ComputeTask) {}
    rpc DimUpdate(Update) returns (NextInstruction) {}
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_SGDServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'ComputeTask': grpc.unary_unary_rpc_method_handler(
          servicer.ComputeTask,
          request_deserializer=sgd__pb2.LWB.FromString,
          response_serializer=sgd__pb2.Update.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'stochastic.SGD', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))