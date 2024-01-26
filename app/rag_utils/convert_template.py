# messages = []
# from jinja2 import Environment, FileSystemLoader

# environment = Environment(loader=FileSystemLoader("./"))
# template = environment.get_template("DetermineReply.jinja2")

# # context = {
# #     "query": "Hi, how are you doing?",
# #     "chat_history": None
# # }

# context = {
# "conversation": "",
# "documentation": "here are my docs 1 2 3",
# "user_query": "actual user quet=y"
# }

# prompt = template.render(context)

# read string in prompt line by line
# for each line, check if it starts with "user" or "assistant"
# if "user", add to messages with role "user"

# prompt = """system:
# s1
# s2

# user:
# u1
# u2
# u3

# assistant:
# a1
# a2
# a3

# system:
# s3
# s4
# s5

# user:
# u4
# """


def convert_jinja_to_messages(jinja_template: str, verbose: bool = False) -> list:
    """
    Converts a Jinja template into a list of messages.

    This function takes a Jinja template as input and converts it into a list of messages. Each message is a dictionary
    with two keys: 'role' and 'content'. 'role' can be one of 'user', 'assistant', or 'system'. 'content' is the 
    corresponding message from the role.

    Args:
        jinja_template (str): The Jinja template to be converted.
        verbose (bool, optional): If True, prints each message. Defaults to False.

    Returns:
        list: A list of dictionaries. Each dictionary represents a message with 'role' and 'content'.
    """
    prompt_lines = jinja_template.split("\n")
    _msgs = ""
    messages = []
    current_message = None

    roles = ["user", "assistant", "system"]

    for line in prompt_lines:
        for role in roles:
            if line.startswith(role):
                if _msgs:
                    messages.append({"role": current_message, "content": _msgs})
                _msgs = ""
                current_message = role
                break
        else:
            _msgs += "\n" + line

    if _msgs != "":
        messages.append({"role": current_message, "content": _msgs})

    if verbose:
        for m in messages:
            print(m)
    return messages
    