# Azure OpenAI demos

## Description

This is a demo showcase of using Azure OpenAI API in simple web app.

Supported scenarios & APIs:

- [Chat](./ChatGPT) simple ChatGPT-like app where you can modfify settings such as `Temperature`, `Model`, `System message`.
- [Chat with file sources](./ChatGPT-LangChain) simple ChatGPT-like where you can add your own file (`.txt`, `.md`)
- [Dall-e V3.0](./Dall-e_3.0) simple image generation application using newest DALL-E 3.0 model

## Installation

> Note!: This repo is based on `openai==1.3.2`. (contains braking changes to 0.x.x version)

In order to run scripts or Web App you must install dependencies:

```sh
pip install -r app/requirements.txt
```

To deploy to Azure Web App (creation is not described):

```sh
az acr build --registry oaimma --resource-group rg-openai-bot --image oimma-streamlit --file WebApp.Dockerfile .
```

## Usage

Simple web app is build upon Streamlit. So to run locally, you can just:

```sh
streamlit run app/app.py
```

## Contributing

Guidelines for contributing to your project.

## License

Information about the project's license.