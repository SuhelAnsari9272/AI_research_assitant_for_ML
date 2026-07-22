def save_graph(graph) : 

    mermaid_code = graph.get_graph().draw_mermaid()

    # print(mermaid_code)
    with open("artifacts/graph.mmd", "w") as f:
        f.write(mermaid_code)


    png_data = graph.get_graph().draw_mermaid_png()

    with open("artifacts/architecture.png", "wb") as f:
        f.write(png_data)

