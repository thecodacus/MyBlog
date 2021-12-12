/**
 * Configure your Gatsby site with this file.
 *
 * See: https://www.gatsbyjs.com/docs/gatsby-config/
 */

module.exports = {
	/* Your site config here */
	plugins: [
		"gatsby-transformer-sharp",
		"gatsby-plugin-sharp",
		"gatsby-plugin-netlify-cms",
		{
			resolve: `gatsby-source-filesystem`,
			options: {
				name: `images`,
				path: `${__dirname}/src/images/`,
			},
		},
		{
			resolve: `gatsby-source-filesystem`,
			options: {
				path: `${__dirname}/static/images/uploads/`,
				name: "media",
			},
		},
		{
			resolve: `gatsby-source-filesystem`,
			options: {
				name: `posts`,
				path: `${__dirname}/src/posts/`,
			},
		},
		{
			resolve: "gatsby-transformer-remark",
			options: {
				plugins: [
					{
						resolve: `gatsby-remark-relative-images`,
						options: {
							name: "media", // Must match the source name ^
							// [Optional] The root of "media_folder" in your config.yml
							// Defaults to "static"
							staticFolderName: "static",
							// [Optional] Include the following fields, use dot notation for nested fields
							// All fields are included by default
							include: ["featuredImage"],
							// [Optional] Exclude the following fields, use dot notation for nested fields
							// No fields are excluded by default
							exclude: [],
						},
					},
					{
						resolve: `gatsby-remark-images`,
						options: {
							maxWidth: 590,
						},
					},
				],
			},
		},
	],
	siteMetadata: {
		title: "The Codacus",
		description: "The Coding Abacus",
		copyright: "Copyright 2021 The Codacus",
	},
}
