"""
APIController
This controller is responsible for handling all API requests and responses.
A single instance of APIController is created and passed to the Project
The API controller has two api endpoint types: 
Commandline API: This is the API that is used to interact with the project from the command line.
Visual API: This is the API that is used to interact with the project from the visual interface.
Response routing takes place in the APIController.
Certian types of responses are routed only to CLI and others to Visual API.

Scratch code to outline structure of the APIController

API modules must be instantiated in the APIController
 - Visual API
 - Commandline API

APIRequest class/object:
This class is used to standardize an API request.
- Request Type: Action, Query

API must load an instance of the projects structure so it knows whats available,
this should still be done through commands that return data and this data will
have to be refreshed by the ui application


API request is recieved
Request is added to the queue
queue is executed one at a time

function handle_request(request_item)
    command = request_item.command # is just the name of the command
    type = request_item.type # is the type of the request
    value = request_item.value # is the value of the request
    block = request_item.block # is the block that the command belongs to

    if request_item.type == "Action":
        if request_item.value is not None:
            response = self.execute_command()
            

        else:
            request_item.block.action()

def execute_command(command, value, block):
    return command.execute(value, block)

I think perhaps all command items get loaded/passed to the API controller


All Project blocks are only exposed externally through their command controller.
Each block can add their own commands, commands there are several types of commands
1.) Action Command: are used to perform an action on the block. 
    If a value is necessary for the action, the request can pass a value to the command 
    Otherwise the command will return a request for user input
2.) Query command: These commands are used to query the block for information. Command can only return a value



"""