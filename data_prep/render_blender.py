import argparse
import os
import sys, json, math
import bpy
import mathutils
import numpy as np

def render_scene(scene_name, n_views, output_folder, color_depth, resolution, translation, rotation, remove_doubles = True, edge_split = True):

    # Set up rendering
    context = bpy.context
    scene = bpy.context.scene
    render = bpy.context.scene.render

    # Turn of color management
    bpy.context.scene.view_settings.view_transform = 'Standard'


    render.engine = 'BLENDER_EEVEE'
    render.image_settings.color_mode = 'RGBA' # ('RGB', 'RGBA', ...)
    render.image_settings.color_depth = color_depth # ('8', '16')
    render.image_settings.file_format = 'PNG'
    render.resolution_x = resolution
    render.resolution_y = resolution
    render.resolution_percentage = 100
    render.film_transparent = True

    scene.use_nodes = True
    scene.view_layers["View Layer"].use_pass_normal = True
    scene.view_layers["View Layer"].use_pass_diffuse_color = True
    scene.view_layers["View Layer"].use_pass_object_index = True

    nodes = scene.node_tree.nodes
    tree = scene.node_tree
    links = tree.links

    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    # Create input render layer node
    render_layers = nodes.new('CompositorNodeRLayers')

    # Create normal output nodes
    scale_node = nodes.new(type="CompositorNodeMixRGB")
    scale_node.blend_type = 'MULTIPLY'
    # scale_node.use_alpha = True
    scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs['Normal'], scale_node.inputs[1])

    bias_node = nodes.new(type="CompositorNodeMixRGB")
    bias_node.blend_type = 'ADD'
    # bias_node.use_alpha = True
    bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_node.outputs[0], bias_node.inputs[1])


    # Delete default cube
    context.active_object.select_set(True)
    bpy.ops.object.delete()

   
    # Import textured mesh
    bpy.ops.object.select_all(action='DESELECT')

    #bpy.ops.import_scene.obj(filepath=args.obj)
    bpy.ops.wm.collada_import(filepath=scene_name)
   

    obj = bpy.context.selected_objects[0]
    obj.location = (0.0, 0.0, 0.0)
    context.view_layer.objects.active = obj

    # # Possibly disable specular shading
    for slot in obj.material_slots:
       node = slot.material.node_tree.nodes['Principled BSDF']
       node.inputs['Specular'].default_value = 0# 0.05

    if remove_doubles:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')
    if edge_split:
        bpy.ops.object.modifier_add(type='EDGE_SPLIT')
        bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
        bpy.ops.object.modifier_apply(modifier="EdgeSplit")

    # Set objekt IDs
    obj.pass_index = 1

    # Make light just directional, disable shadows.
    light = bpy.data.lights['Light']
    light.type = 'SUN'
    light.use_shadow = False
    # Possibly disable specular shading:
    light.specular_factor = 0 # 1.0
    light.energy = 1.0 # 10

    # Add another light source so stuff facing away from light is not completely dark
    bpy.ops.object.light_add(type='SUN')
    light2 = bpy.data.lights['Sun']
    light2.use_shadow = False
    light2.specular_factor = 0 #1.0
    light2.energy = 1.0 #0.015
    bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Light'].rotation_euler
    bpy.data.objects['Sun'].rotation_euler[0] += 180

    # Place camera
    cam = scene.objects['Camera']
    cam.location = (0, 2, 0) # put it a bit further away to see the effect of translations better
    cam.data.lens = 35
    cam.data.sensor_width = 32

    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'

    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    cam.parent = cam_empty

    scene.collection.objects.link(cam_empty)
    context.view_layer.objects.active = cam_empty
    cam_constraint.target = cam_empty
    ########################################
    # Configure Camera Background
    filepath = "background.png"

    # Displaying the Background in the camera view
    img = bpy.data.images.load(filepath)
    cam.data.show_background_images = True
    bg = cam.data.background_images.new()
    bg.image = img
    bpy.context.scene.render.film_transparent = True

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    for every_node in tree.nodes:
        tree.nodes.remove(every_node)

    RenderLayers_node = tree.nodes.new('CompositorNodeRLayers')   
    #RenderLayers_node.location = -300,300

    comp_node = tree.nodes.new('CompositorNodeComposite')   
    #comp_node.location = 400,300

    AplhaOver_node = tree.nodes.new(type="CompositorNodeAlphaOver")
    #AplhaOver_node.location = 150,450


    Scale_node = tree.nodes.new(type="CompositorNodeScale")
    bpy.data.scenes["Scene"].node_tree.nodes["Scale"].space = 'RENDER_SIZE'
    #Scale_node.location = -150,500

    Image_node = tree.nodes.new(type="CompositorNodeImage")
    Image_node.image = img  
    #Image_node.location = -550,500

    links = tree.links
    link1 = links.new(RenderLayers_node.outputs[0], AplhaOver_node.inputs[2])
    link2 = links.new(AplhaOver_node.outputs[0], comp_node.inputs[0])
    link3 = links.new(Scale_node.outputs[0], AplhaOver_node.inputs[1])
    link4 = links.new(Image_node.outputs[0], Scale_node.inputs[0])



    #######################################33

    model_identifier = os.path.split(os.path.split(scene_name)[1])[1]
    fp = os.path.join(os.path.abspath(output_folder), model_identifier)

    render_info = {}

    # Trying to center the object
    # me = obj.data
    # mw = obj.matrix_world
    # origin = sum((v.co for v in me.vertices), mathutils.Vector()) / len(me.vertices)

    # T = mathutils.Matrix.Translation(-origin)
    # me.transform(T)
    # mw.translation = mw @ origin

    # For now, do it manually
    obj.location = (-0.25, 0.5, 0)
    cam_empty.rotation_euler = (0.0, 0.0, 0.0)
    x_rot = 0.0
    z_rot = 0.0

    for i in range(n_views):
        render_file_path = fp + '/{0:05d}'.format(int(i))
        info_view = {"azimuth": z_rot,  "elevation" : x_rot, "x" : obj.location[0], "y" : obj.location[1], "z" : obj.location[2]}
        render_info['{0:05d}'.format(int(i))] = info_view 
        scene.render.filepath = render_file_path

        # change the viewpoint for the next image
        if translation:
            # CHange object loc 
            obj.location = (np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5))
        
        if rotation: 
            # change camera rotation
            x_rot = np.random.uniform(0, 2*math.pi) # elevation
            z_rot = np.random.uniform(0, 2*math.pi) # azimuth
            cam_empty.rotation_euler = (x_rot, 0.0, z_rot)
    
        bpy.ops.render.render(write_still=True)  # render still

        # For debugging the workflow
    # bpy.ops.wm.save_as_mainfile(filepath='end.blend')

    with open(fp +"/render_params.json", "w") as f:
        json.dump(render_info, f)

    bpy.data.objects.remove(obj, do_unlink=True)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create data for')
    parser.add_argument('--n_images', type=int, default=50,
                        help='number of views/images to be rendered per scende')
    parser.add_argument('--scene_name', type=str,
                        help='Path to the .dae object')
    parser.add_argument('--scale', type=float, default=1,
                        help='Scaling factor applied to model. Depends on size of mesh.') # ?????
    parser.add_argument('--output_folder', type=str, default="output/rot_dataset/",
                        help='Scaling that is applied to depth. Depends on size of mesh.') # ????
    parser.add_argument('--color_depth', type=str, default='8',
                        help='Number of bit per channel used for output. Either 8 or 16.') # ????
    parser.add_argument('--resolution', type=int, default=128, # 128
                        help='Resolution of the images.')
    parser.add_argument('--translation', action='store_true')
    parser.add_argument('--rotation', action='store_true')

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)


    render_scene(args.scene_name, args.n_images, args.output_folder, args.color_depth, args.resolution, args.translation, args.rotation)
        
        