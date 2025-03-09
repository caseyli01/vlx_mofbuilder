def gro_show(gro_file, w=800, h=600, res_id=True, res_name=True):
    try:
        import py3Dmol

        viewer = py3Dmol.view(width=w, height=h)
        with open(gro_file, "r") as f:
            lines = f.readlines()

        viewer.addModel("".join(lines), "gro")
        viewer.setStyle({"stick": {}})

        viewer.setViewStyle({"style": "outline", "width": 0.05})
        viewer.setStyle({"stick": {}, "sphere": {"scale": 0.20}})
        if res_id or res_name:
            for i in range(2, len(lines) - 1):
                if lines[i].strip() == "":
                    continue
                if lines[i - 1][0:5] == lines[i][0:5]:
                    continue

                value_resnumber = int((lines[i])[0:5])
                value_resname = lines[i][5:10]
                if value_resname.strip() == "TNO":
                    continue
                # value_label = lines[i][10:15]
                # value_atom_number = int(lines[i][15:20])
                value_x = float(lines[i][20:28]) * 10  # x
                value_y = float(lines[i][28:36]) * 10  # y
                value_z = float(lines[i][36:44]) * 10  # z

                text = ""
                if res_name:
                    text += str(value_resname)
                if res_id:
                    text += str(value_resnumber)

                viewer.addLabel(
                    text,
                    {
                        "position": {
                            "x": value_x,
                            "y": value_y,
                            "z": value_z,
                        },
                        "alignment": "center",
                        "fontColor": "white",
                        "font": "Arial",
                        "fontSize": 5,
                        "backgroundColor": "black",
                        "backgroundOpacity": 0.5,
                    },
                )
        viewer.render()
        viewer.zoomTo()
        viewer.show()
    except ImportError:
        raise ImportError("Unable to import py3Dmol")
